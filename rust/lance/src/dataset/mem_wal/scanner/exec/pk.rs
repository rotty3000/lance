// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared primary-key helpers for the LSM scanner execution nodes.
//!
//! Centralizes PK column resolution and per-row hashing so that every
//! consumer (e.g. [`super::PkBlockFilterExec`])
//! resolves and hashes a primary key the same way. The row hash is kept
//! consistent with the variants supported by [`super::compute_pk_hash_from_scalars`]
//! so a single PK produces the same hash regardless of which exec consumes it.

use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, Schema};
use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result as DFResult};
use lance_core::{Error, Result};

/// Column name for a row address (the in-source row offset).
pub const ROW_ADDRESS_COLUMN: &str = "_rowaddr";

/// Resolve the column index of each primary-key column in `batch`.
pub fn resolve_pk_indices(batch: &RecordBatch, pk_columns: &[String]) -> DFResult<Vec<usize>> {
    pk_columns
        .iter()
        .map(|col| {
            batch
                .schema()
                .column_with_name(col)
                .map(|(idx, _)| idx)
                .ok_or_else(|| {
                    DataFusionError::Internal(format!("Primary key column '{}' not found", col))
                })
        })
        .collect()
}

/// Primary-key column types we can hash exactly in the fast path.
///
/// Anything else is rejected by [`validate_pk_types`] at the scanner boundary,
/// so the hot hash path never silently collapses distinct keys to one hash
/// (which would over-block in the block-list and drop live rows).
///
/// FORK PATCH (knowdb): upstream this list omits `FixedSizeBinary`, `Date32`
/// and `Date64` — through v11.0.0-beta.20 and `main`, not just the 10.0.0 we
/// pin, so there is no release to upgrade to.
///
/// The omission is a stale inventory, not a policy. This list is exactly the
/// set of types that got explicit downcast arms in `2caaa0c20` (#6929,
/// 2026-05-26), a ~2,300-line stale-read refactor whose message never mentions
/// PK validation. One week later `dd25f21d0` (#7011) added a *second*
/// primary-key type allow-list to the same subsystem —
/// [`crate::dataset::mem_wal::index::is_encodable_pk_type`], which gates the
/// same `pk_columns` — and that one admits all three, naming
/// `FixedSizeBinary(16)`/UUID as its motivating key shape. Nothing
/// back-propagated here, so mem_wal now accepts as a composite primary key
/// what its own scanner then refuses to hash; `every_composite_encodable_pk_type_is_also_hashable`
/// below pins that invariant.
///
/// Two further reasons the exclusion has no defender: the format spec's
/// primary-key rules (non-nullable, leaf, not inside a list or map, per
/// `Schema::verify_primary_key`) impose no type allow-list at all, so a table
/// keyed on `FixedSizeBinary` creates successfully and then fails to scan; and
/// the doc comment's stated hazard — distinct keys collapsing to one hash — was
/// already closed by the value-distinguishing `ScalarValue` fallback that
/// #6929 added in the same commit. The only exclusion upstream has ever pinned
/// with a reason is `Float64` (NaN / `-0.0` make equality unsound).
///
/// Drop this patch once upstream accepts the equivalent change.
pub fn is_supported_pk_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Date32
            | DataType::Date64
            | DataType::Boolean
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::FixedSizeBinary(_)
    )
}

/// Validate that every primary-key column has a type we can hash exactly.
///
/// Rejects unsupported types with a descriptive error at the API boundary
/// rather than degrading to a constant hash. Call this where a scanner that
/// hashes primary keys is built.
pub fn validate_pk_types(schema: &Schema, pk_columns: &[String]) -> Result<()> {
    for col in pk_columns {
        let field = schema.field_with_name(col).map_err(|_| {
            Error::invalid_input(format!("Primary key column '{}' not found in schema", col))
        })?;
        if !is_supported_pk_type(field.data_type()) {
            return Err(Error::invalid_input(format!(
                "Primary key column '{}' has unsupported type {:?} for hashing; supported types: \
                 Int8/16/32/64, UInt8/16/32/64, Date32/Date64, Boolean, Utf8/LargeUtf8, \
                 Binary/LargeBinary/FixedSizeBinary",
                col,
                field.data_type()
            )));
        }
    }
    Ok(())
}

/// Hash a single row's primary key, identified by the `pk_indices` column
/// positions and `row_idx`.
///
/// Must stay byte-for-byte consistent with
/// [`super::compute_pk_hash_from_scalars`] so a single PK hashes the same
/// regardless of which exec consumes it. Supported types are validated up
/// front by [`validate_pk_types`]; the trailing branch is a defensive,
/// value-distinguishing fallback that should be unreachable in validated plans.
pub fn compute_pk_hash(batch: &RecordBatch, pk_indices: &[usize], row_idx: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for &col_idx in pk_indices {
        let col = batch.column(col_idx);
        let is_null = col.is_null(row_idx);
        is_null.hash(&mut hasher);

        if !is_null {
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int8Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int16Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt8Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt16Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::UInt64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Date32Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::Date64Array>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::BooleanArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::BinaryArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<arrow_array::LargeBinaryArray>() {
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
            {
                // Hashed as a plain byte slice, exactly like `BinaryArray`, so a
                // key hashes the same whichever width the column declares.
                arr.value(row_idx).hash(&mut hasher);
            } else if let Ok(scalar) = ScalarValue::try_from_array(col.as_ref(), row_idx) {
                // Defensive fallback: distinguish by value rather than collapse.
                format!("{:?}", scalar).hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::index::is_encodable_pk_type;
    use arrow_array::{Date32Array, Date64Array, FixedSizeBinaryArray};
    use arrow_schema::Field;
    use std::sync::Arc;

    const HASH_WIDTH: i32 = 32;

    fn fixed_size_binary_schema() -> Schema {
        Schema::new(vec![Field::new(
            "content_hash",
            DataType::FixedSizeBinary(HASH_WIDTH),
            false,
        )])
    }

    fn batch_of(values: &[[u8; 32]]) -> RecordBatch {
        let array = FixedSizeBinaryArray::try_from_iter(values.iter()).unwrap();
        RecordBatch::try_new(Arc::new(fixed_size_binary_schema()), vec![Arc::new(array)]).unwrap()
    }

    /// A content-addressed key is the canonical use of a fixed-width binary
    /// column, and the format spec's primary-key rules (non-nullable, leaf,
    /// not inside a list or map) admit it.
    #[test]
    fn validate_pk_types_accepts_fixed_size_binary() {
        validate_pk_types(&fixed_size_binary_schema(), &["content_hash".to_string()])
            .expect("FixedSizeBinary must be usable as a primary key");
    }

    /// `compute_pk_hash` and `compute_pk_hash_from_scalars` must agree, or a
    /// key hashes differently depending on which exec consumes it.
    #[test]
    fn fixed_size_binary_hashes_agree_across_both_pk_hash_paths() {
        let value = [7u8; 32];
        let batch = batch_of(&[value]);

        let from_array = compute_pk_hash(&batch, &[0], 0);
        let from_scalars =
            super::super::compute_pk_hash_from_scalars(&[ScalarValue::FixedSizeBinary(
                HASH_WIDTH,
                Some(value.to_vec()),
            )]);

        assert_eq!(
            from_array, from_scalars,
            "array and scalar hashing of the same FixedSizeBinary key diverged"
        );
    }

    /// The hazard `validate_pk_types` exists to prevent: distinct keys
    /// collapsing to one hash would over-block the block-list and drop live
    /// rows.
    #[test]
    fn distinct_fixed_size_binary_keys_hash_differently() {
        let batch = batch_of(&[[1u8; 32], [2u8; 32]]);

        assert_ne!(
            compute_pk_hash(&batch, &[0], 0),
            compute_pk_hash(&batch, &[0], 1),
            "distinct FixedSizeBinary keys collapsed to one hash"
        );
    }

    /// The invariant this module's allow-list drifted away from: a type that
    /// `validate_mem_index_config` will accept as a *composite* primary-key
    /// column must also be hashable here, or a shard whose PK index builds
    /// successfully cannot be scanned. The implication is one-way — a type may
    /// be hashable without having an order-preserving key encoding.
    #[test]
    fn every_composite_encodable_pk_type_is_also_hashable() {
        let types = [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Date32,
            DataType::Date64,
            DataType::Boolean,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::FixedSizeBinary(16),
        ];
        for data_type in types {
            if is_encodable_pk_type(&data_type) {
                assert!(
                    is_supported_pk_type(&data_type),
                    "{data_type:?} encodes into a composite primary key but cannot be hashed \
                     as one, so a shard keyed on it builds and then fails to scan"
                );
            }
        }
    }

    fn date_schema(name: &str, data_type: DataType) -> Schema {
        Schema::new(vec![Field::new(name, data_type, false)])
    }

    /// `is_encodable_pk_type` has admitted `Date32`/`Date64` as composite
    /// primary-key columns since #7011; this list never caught up.
    #[test]
    fn validate_pk_types_accepts_date32_and_date64() {
        validate_pk_types(&date_schema("day", DataType::Date32), &["day".to_string()])
            .expect("Date32 must be usable as a primary key");
        validate_pk_types(
            &date_schema("instant", DataType::Date64),
            &["instant".to_string()],
        )
        .expect("Date64 must be usable as a primary key");
    }

    /// Same agreement requirement as the `FixedSizeBinary` case: a typed array
    /// arm without its matching `ScalarValue` arm hashes the same key two
    /// different ways.
    #[test]
    fn date32_hashes_agree_across_both_pk_hash_paths() {
        let day = 20_000_i32;
        let batch = RecordBatch::try_new(
            Arc::new(date_schema("day", DataType::Date32)),
            vec![Arc::new(Date32Array::from(vec![day]))],
        )
        .unwrap();

        assert_eq!(
            compute_pk_hash(&batch, &[0], 0),
            super::super::compute_pk_hash_from_scalars(&[ScalarValue::Date32(Some(day))]),
            "array and scalar hashing of the same Date32 key diverged"
        );
    }

    #[test]
    fn date64_hashes_agree_across_both_pk_hash_paths() {
        let instant = 1_700_000_000_000_i64;
        let batch = RecordBatch::try_new(
            Arc::new(date_schema("instant", DataType::Date64)),
            vec![Arc::new(Date64Array::from(vec![instant]))],
        )
        .unwrap();

        assert_eq!(
            compute_pk_hash(&batch, &[0], 0),
            super::super::compute_pk_hash_from_scalars(&[ScalarValue::Date64(Some(instant))]),
            "array and scalar hashing of the same Date64 key diverged"
        );
    }

    #[test]
    fn distinct_date32_keys_hash_differently() {
        let batch = RecordBatch::try_new(
            Arc::new(date_schema("day", DataType::Date32)),
            vec![Arc::new(Date32Array::from(vec![20_000, 20_001]))],
        )
        .unwrap();

        assert_ne!(
            compute_pk_hash(&batch, &[0], 0),
            compute_pk_hash(&batch, &[0], 1),
            "distinct Date32 keys collapsed to one hash"
        );
    }
}
