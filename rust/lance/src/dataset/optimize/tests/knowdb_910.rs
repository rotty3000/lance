// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! knowdb #910 verification: content compaction must be able to rewrite
//! fragments that are NOT adjacent in the manifest (the hot fragments between
//! them are excluded), and a scalar index append must commit beside a
//! concurrent UPDATE of the same fragments.

use super::*;
use crate::dataset::UpdateBuilder;
use arrow_array::StringArray;

const ROWS_PER_FRAG: i64 = 100;
const FRAGS: i64 = 6;

fn batch(start: i64, n: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(start..start + n)),
            Arc::new(Int64Array::from_iter_values(
                (start..start + n).map(|v| v * 10),
            )),
            Arc::new(StringArray::from_iter_values(
                (start..start + n).map(|_| "pending"),
            )),
        ],
    )
    .unwrap()
}

fn write_params() -> WriteParams {
    WriteParams {
        max_rows_per_file: ROWS_PER_FRAG as usize,
        max_rows_per_group: ROWS_PER_FRAG as usize,
        data_storage_version: Some(LanceFileVersion::Stable),
        ..Default::default()
    }
}

async fn write_frags(uri: &str, start: i64, frags: i64, mode: WriteMode) -> Dataset {
    let data = batch(start, ROWS_PER_FRAG * frags);
    let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
    let params = WriteParams {
        mode,
        ..write_params()
    };
    Dataset::write(reader, uri, Some(params)).await.unwrap()
}

/// `(a, b)` of every row the `a = v` filter returns.
async fn rows_where_a(ds: &Dataset, v: i64) -> Vec<(i64, i64)> {
    let mut scan = ds.scan();
    scan.filter(&format!("a = {v}"))
        .unwrap()
        .project(&["a", "b"])
        .unwrap();
    let batch = scan.try_into_batch().await.unwrap();
    let a = batch["a"].as_any().downcast_ref::<Int64Array>().unwrap();
    let b = batch["b"].as_any().downcast_ref::<Int64Array>().unwrap();
    (0..batch.num_rows())
        .map(|i| (a.value(i), b.value(i)))
        .collect()
}

async fn assert_index_answers_a_eq(ds: &Dataset) {
    let mut scan = ds.scan();
    scan.filter("a = 1").unwrap();
    let plan = scan.explain_plan(true).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery"),
        "the a = v filter must be answered by the index: {plan}"
    );
}

/// One compaction task over fragments 1 and 4 (two hot fragments between
/// them), listed in id order, through the re-encode path (knowdb's) and the
/// binary-copy path. Every `a = v` lookup must return exactly its own row.
///
/// ⚠ The task must list its fragments in ASCENDING id order: the remap pairs
/// the old row addresses (a `RoaringTreemap`, iterated ascending) with the new
/// rows in WRITE order. The re-encode path refuses a descending task
/// ([`a_descending_task_is_refused_by_the_reencode_path_910`]); the
/// binary-copy path does not check, and with `IndexRemapMode::Direct` (the
/// default) it remaps every row to the wrong address (`a = 100` answered row
/// 400). Lance's own planner never builds such a task, because the manifest
/// stays id-sorted ([`the_manifest_stays_id_sorted_after_a_splice_910`]).
#[rstest]
#[tokio::test]
async fn a_task_over_non_adjacent_fragments_keeps_the_index_exact_910(
    #[values(false, true)] binary_copy: bool,
) {
    let dir = TempStrDir::default();
    let mut ds = write_frags(&dir, 0, FRAGS, WriteMode::Create).await;
    create_scalar_index(&mut ds, "a", false).await;
    // Holes in the remap. Binary copy refuses a fragment with a deletion file.
    let deleted = |v: i64| !binary_copy && (400..500).contains(&v) && v % 7 == 0;
    if !binary_copy {
        ds.delete("a >= 400 AND a < 500 AND a % 7 = 0")
            .await
            .unwrap();
    }

    let picked: Vec<Fragment> = [1usize, 4]
        .iter()
        .map(|&i| ds.manifest.fragments[i].clone())
        .collect();
    assert_eq!(picked.iter().map(|f| f.id).collect::<Vec<_>>(), vec![1, 4]);
    let options = CompactionOptions {
        enable_binary_copy: binary_copy,
        enable_binary_copy_force: binary_copy,
        ..Default::default()
    };
    let result = rewrite_files(Cow::Borrowed(&ds), TaskData { fragments: picked }, &options)
        .await
        .unwrap();
    assert_eq!(result.new_fragments.len(), 1);
    commit_compaction(
        &mut ds,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();

    let ids: Vec<u64> = ds.manifest.fragments.iter().map(|f| f.id).collect();
    assert!(
        !ids.contains(&1) && !ids.contains(&4),
        "fragments 1 and 4 are replaced: {ids:?}"
    );
    assert_eq!(ids.len(), FRAGS as usize - 1, "{ids:?}");
    let new_id = *ids.iter().max().unwrap();
    let index = ds
        .load_indices()
        .await
        .unwrap()
        .iter()
        .find(|i| i.name == "scalar")
        .cloned()
        .unwrap();
    let coverage = index.fragment_bitmap.unwrap();
    assert!(
        coverage.contains(new_id as u32),
        "the index covers the output: {coverage:?}"
    );

    assert_index_answers_a_eq(&ds).await;
    for v in 0..ROWS_PER_FRAG * FRAGS {
        let expected = if deleted(v) {
            vec![]
        } else {
            vec![(v, v * 10)]
        };
        assert_eq!(rows_where_a(&ds, v).await, expected, "a = {v}");
    }
}

/// The #910 B race: a content index append (`optimize_indices(append)`, a
/// `CreateIndex`) and a worker's status UPDATE start from the SAME read
/// version and touch the same fragments (indexed and unindexed). Whichever
/// commits second must still commit, with NO retry of its own, and both
/// effects must be visible.
#[rstest]
#[tokio::test]
async fn an_index_append_and_an_update_of_the_same_fragments_both_commit_910(
    #[values(false, true)] update_first: bool,
) {
    let dir = TempStrDir::default();
    let mut ds = write_frags(&dir, 0, 3, WriteMode::Create).await;
    create_scalar_index(&mut ds, "a", false).await;
    // Three unindexed fragments: the tail an append covers.
    let mut ds = {
        let data = batch(300, 300);
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        ds.append(reader, Some(write_params())).await.unwrap();
        ds
    };
    let read_version = ds.version().version;

    // One row in an indexed fragment (1), one in an unindexed one (4).
    let update = |ds: Dataset| async move {
        UpdateBuilder::new(Arc::new(ds))
            .update_where("(a = 150 OR a = 450) AND s = 'pending'")
            .unwrap()
            .set("s", "'done'")
            .unwrap()
            .conflict_retries(0)
            .build()
            .unwrap()
            .execute()
            .await
    };
    let append = |mut ds: Dataset| async move {
        ds.optimize_indices(&lance_index::optimize::OptimizeOptions::append())
            .await
            .map(|_| ds)
    };

    let stale = ds.checkout_version(read_version).await.unwrap();
    if update_first {
        let updated = update(ds.clone()).await.unwrap();
        assert_eq!(updated.rows_updated, 2);
        append(stale)
            .await
            .expect("an index append from the pre-update version must commit");
    } else {
        append(ds.clone()).await.unwrap();
        let updated = update(stale)
            .await
            .expect("an update from the pre-append version must commit without a retry");
        assert_eq!(updated.rows_updated, 2);
    }

    let mut ds = ds;
    ds.checkout_latest().await.unwrap();
    assert_eq!(ds.count_rows(None).await.unwrap(), 600);
    assert_eq!(
        ds.count_rows(Some("s = 'done'".into())).await.unwrap(),
        2,
        "the update landed"
    );
    let index = ds
        .load_indices()
        .await
        .unwrap()
        .iter()
        .filter(|i| i.name == "scalar")
        .cloned()
        .collect::<Vec<_>>();
    let covered: RoaringBitmap = index
        .iter()
        .filter_map(|i| i.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |acc, b| acc | b);
    for id in 3..6u32 {
        assert!(
            covered.contains(id),
            "the append covers appended fragment {id}: {covered:?}"
        );
    }
    assert_index_answers_a_eq(&ds).await;
    for v in [150, 450, 151, 451, 0, 599] {
        assert_eq!(rows_where_a(&ds, v).await, vec![(v, v * 10)], "a = {v}");
    }
}

/// The re-encode path's row-address capture assumes ascending addresses and
/// fails the task (nothing is committed) when the fragments run high → low.
#[tokio::test]
async fn a_descending_task_is_refused_by_the_reencode_path_910() {
    let dir = TempStrDir::default();
    let mut ds = write_frags(&dir, 0, FRAGS, WriteMode::Create).await;
    create_scalar_index(&mut ds, "a", false).await;
    let picked: Vec<Fragment> = [4usize, 1]
        .iter()
        .map(|&i| ds.manifest.fragments[i].clone())
        .collect();
    let err = rewrite_files(
        Cow::Borrowed(&ds),
        TaskData { fragments: picked },
        &CompactionOptions::default(),
    )
    .await
    .expect_err("a descending task must not rewrite");
    assert!(err.to_string().contains("ordered"), "{err}");
}

/// A contiguous rewrite's output takes a NEW, higher id, and the committed
/// manifest is still id-sorted, so lance's own planner only ever plans tasks
/// in ascending id order.
#[tokio::test]
async fn the_manifest_stays_id_sorted_after_a_splice_910() {
    let dir = TempStrDir::default();
    let mut ds = write_frags(&dir, 0, 4, WriteMode::Create).await;
    create_scalar_index(&mut ds, "a", false).await;
    let picked: Vec<Fragment> = ds.manifest.fragments[1..3].to_vec();
    let options = CompactionOptions::default();
    let result = rewrite_files(Cow::Borrowed(&ds), TaskData { fragments: picked }, &options)
        .await
        .unwrap();
    commit_compaction(
        &mut ds,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await
    .unwrap();
    let ids: Vec<u64> = ds.manifest.fragments.iter().map(|f| f.id).collect();
    assert_eq!(ids, vec![0, 3, 4]);

    let plan = plan_compaction(&ds, &options).await.unwrap();
    let planned: Vec<Vec<u64>> = plan
        .tasks
        .iter()
        .map(|t| t.fragments.iter().map(|f| f.id).collect())
        .collect();
    assert_eq!(planned, vec![vec![0, 3, 4]]);
    compact_files(&mut ds, options, None).await.unwrap();
    for v in 0..400 {
        assert_eq!(rows_where_a(&ds, v).await, vec![(v, v * 10)], "a = {v}");
    }
}
