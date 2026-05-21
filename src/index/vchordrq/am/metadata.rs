use pgrx::pg_sys;
use std::ffi::CStr;

pub const FEED_ID_META_HASH: &str = "feed_id_meta_hash";
pub const STATUS_META: &str = "status_meta";
pub const VISIBILITY_META: &str = "visibility_meta";
pub const DELETED_META: &str = "deleted_meta";
pub const ELIGIBILITY_FLAGS_META: &str = "eligibility_flags_meta";
pub const GEO_CELL_META: &str = "geo_cell_meta";
pub const CREATED_AT_BUCKET_META: &str = "created_at_bucket_meta";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MetadataColumnKind {
    Feed,
    Flags,
    Status,
    Deleted,
    Visibility,
    Geo,
    Time,
    Other,
}

impl MetadataColumnKind {
    pub fn from_name(name: &str) -> Self {
        match name {
            FEED_ID_META_HASH => Self::Feed,
            ELIGIBILITY_FLAGS_META => Self::Flags,
            STATUS_META => Self::Status,
            DELETED_META => Self::Deleted,
            VISIBILITY_META => Self::Visibility,
            GEO_CELL_META => Self::Geo,
            CREATED_AT_BUCKET_META => Self::Time,
            _ => Self::Other,
        }
    }

    pub const fn active_name(self) -> &'static str {
        match self {
            Self::Feed => "feed",
            Self::Flags => "flags",
            Self::Status => "status",
            Self::Deleted => "deleted",
            Self::Visibility => "visibility",
            Self::Geo => "geo",
            Self::Time => "time",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetadataColumn {
    pub name: String,
    pub kind: MetadataColumnKind,
    pub index_attno: usize,
    pub metadata_index: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MetadataSchema {
    columns: Vec<MetadataColumn>,
}

impl MetadataSchema {
    pub fn cols(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[MetadataColumn] {
        &self.columns
    }

    pub fn by_name(&self, name: &str) -> Option<&MetadataColumn> {
        self.columns.iter().find(|column| column.name == name)
    }
}

pub unsafe fn detect_schema(index_relation: pg_sys::Relation) -> MetadataSchema {
    unsafe {
        let index = (*index_relation).rd_index;
        if index.is_null() {
            return MetadataSchema::default();
        }
        if (*index).indnkeyatts != 1 {
            pgrx::error!("vchordrq requires exactly one key column");
        }
        if (*index).indnatts < 1 {
            pgrx::error!("vchordrq requires one vector key column");
        }
        if (*index).indnatts as usize > 1 + vchordrq::MAX_METADATA_ATTRS {
            pgrx::error!(
                "vchordrq metadata supports at most {} INCLUDE columns",
                vchordrq::MAX_METADATA_ATTRS
            );
        }
        if (*index).indnatts == 1 {
            return MetadataSchema::default();
        }
        let atts = index_attrs(index_relation);
        let mut columns = Vec::with_capacity((*index).indnatts as usize - 1);
        for index_attno in 1..(*index).indnatts as usize {
            let Some(att) = atts.get(index_attno) else {
                pgrx::error!("vchordrq index metadata catalog is inconsistent");
            };
            if att.atttypid != pg_sys::INT8OID {
                pgrx::error!("vchordrq metadata INCLUDE columns must be bigint");
            }
            let name = CStr::from_ptr(att.attname.data.as_ptr())
                .to_str()
                .unwrap_or_default()
                .to_owned();
            columns.push(MetadataColumn {
                kind: MetadataColumnKind::from_name(&name),
                name,
                index_attno,
                metadata_index: index_attno - 1,
            });
        }
        MetadataSchema { columns }
    }
}

pub unsafe fn metadata_from_index_values(
    values: *const pg_sys::Datum,
    is_null: *const bool,
    schema: &MetadataSchema,
) -> vchordrq::CandidateMetadata {
    unsafe {
        let mut metadata = vchordrq::CandidateMetadata::default();
        for column in schema.columns() {
            if is_null.add(column.index_attno).read() {
                continue;
            }
            metadata.set(
                column.metadata_index,
                values.add(column.index_attno).read().value() as i64,
            );
        }
        metadata
    }
}

unsafe fn index_attrs(
    index_relation: pg_sys::Relation,
) -> &'static [pg_sys::FormData_pg_attribute] {
    unsafe {
        let att = &mut *(*index_relation).rd_att;
        #[cfg(any(feature = "pg14", feature = "pg15", feature = "pg16", feature = "pg17"))]
        {
            att.attrs.as_slice(att.natts as _)
        }
        #[cfg(feature = "pg18")]
        {
            let ptr = att
                .compact_attrs
                .as_ptr()
                .add(att.natts as _)
                .cast::<pg_sys::FormData_pg_attribute>();
            std::slice::from_raw_parts(ptr, att.natts as _)
        }
    }
}
