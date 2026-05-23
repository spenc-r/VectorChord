use pgrx::pg_sys;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use validator::Validate;

use super::Reloption;
use crate::index::vchordrq::types::{VchordrqIndexingOptions, VchordrqMetadataColumnOp};

pub type MetadataColumnOp = VchordrqMetadataColumnOp;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataColumnSemantics {
    ops: BTreeSet<MetadataColumnOp>,
    pub exact: bool,
}

impl MetadataColumnSemantics {
    pub fn new(ops: impl IntoIterator<Item = MetadataColumnOp>, exact: bool) -> Self {
        Self {
            ops: ops.into_iter().collect(),
            exact,
        }
    }

    pub fn supports(&self, op: MetadataColumnOp) -> bool {
        self.ops.contains(&op)
    }
}

#[derive(Debug, Clone)]
pub struct MetadataColumn {
    pub name: String,
    pub semantics: Option<MetadataColumnSemantics>,
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

    pub fn declared_by_name(&self, name: &str) -> Option<&MetadataColumn> {
        self.columns
            .iter()
            .find(|column| column.name == name && column.semantics.is_some())
    }

    #[cfg(test)]
    pub fn new_for_test(columns: Vec<MetadataColumn>) -> Self {
        Self { columns }
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
        let declared = declared_metadata(index_relation);
        if (*index).indnatts == 1 {
            if let Some(name) = declared.keys().next() {
                pgrx::error!(
                    "vchordrq declared metadata column `{}` must be a bigint INCLUDE column",
                    name
                );
            }
            return MetadataSchema::default();
        }
        let atts = index_attrs(index_relation);
        let mut seen_include_names = BTreeSet::new();
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
            seen_include_names.insert(name.clone());
            columns.push(MetadataColumn {
                semantics: declared.get(&name).cloned(),
                name,
                index_attno,
                metadata_index: index_attno - 1,
            });
        }
        for name in declared.keys() {
            if !seen_include_names.contains(name) {
                pgrx::error!(
                    "vchordrq declared metadata column `{}` must be a bigint INCLUDE column",
                    name
                );
            }
        }
        MetadataSchema { columns }
    }
}

unsafe fn declared_metadata(
    index_relation: pg_sys::Relation,
) -> BTreeMap<String, MetadataColumnSemantics> {
    unsafe {
        let reloption = (*index_relation).rd_options as *const Reloption;
        let s = Reloption::options(reloption, c"").to_string_lossy();
        let options = match toml::from_str::<VchordrqIndexingOptions>(&s) {
            Ok(options) => options,
            Err(error) => pgrx::error!("failed to parse options: {}", error),
        };
        if let Err(errors) = Validate::validate(&options) {
            pgrx::error!("failed to validate options: {errors}");
        }
        options
            .metadata
            .columns
            .into_iter()
            .map(|column| {
                let semantics = MetadataColumnSemantics::new(column.ops, column.exact);
                (column.name, semantics)
            })
            .collect()
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
