// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute
// this software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025-2026 TensorChord Inc.

use pgrx::pg_sys;
use std::collections::BTreeSet;
use std::ffi::CStr;
use std::ptr;
use std::ptr::NonNull;

use super::metadata::{MetadataColumnOp, MetadataColumnSemantics, MetadataSchema};

#[derive(Debug, Clone, Default)]
pub struct CompiledMetadataQual {
    pub predicates: Vec<MetadataPredicate>,
    pub supported_qual_count: usize,
    pub unsupported_qual_count: usize,
    pub all_quals_covered: bool,
    pub unavailable_param_count: usize,
}

#[derive(Debug, Clone)]
pub struct MetadataPredicate {
    pub metadata_index: usize,
    pub column_name: String,
    pub exact: bool,
    pub op: MetadataPredicateOp,
}

#[derive(Debug, Clone)]
pub enum MetadataPredicateOp {
    Eq(i64),
    In(Vec<i64>),
    Ge(i64),
    Gt(i64),
    Le(i64),
    Lt(i64),
    BitmaskContains(i64),
}

impl MetadataPredicate {
    pub fn is_definitely_false(&self, metadata: vchordrq::CandidateMetadata) -> Option<bool> {
        let value = metadata.get(self.metadata_index)?;
        Some(match &self.op {
            MetadataPredicateOp::Eq(target) => value != *target,
            MetadataPredicateOp::In(targets) => !targets.contains(&value),
            MetadataPredicateOp::Ge(target) => value < *target,
            MetadataPredicateOp::Gt(target) => value <= *target,
            MetadataPredicateOp::Le(target) => value > *target,
            MetadataPredicateOp::Lt(target) => value >= *target,
            MetadataPredicateOp::BitmaskContains(mask) => (value & *mask) != *mask,
        })
    }

    pub const fn is_exact_for_heap_skip(&self) -> bool {
        self.exact
    }
}

#[derive(Debug, Clone)]
pub struct MetadataActiveColumns {
    all: bool,
    active: BTreeSet<String>,
}

impl MetadataActiveColumns {
    pub fn parse(value: &str, schema: &MetadataSchema) -> Self {
        let mut active = BTreeSet::new();
        let mut all = false;
        let mut saw_column_token = false;
        for raw in value.split(',') {
            let token = raw.trim();
            if token.is_empty() {
                continue;
            }
            if token.eq_ignore_ascii_case("all") || token == "*" {
                all = true;
                continue;
            }
            saw_column_token = true;
            if schema.declared_by_name(token).is_some() {
                active.insert(token.to_owned());
            }
        }
        if !saw_column_token {
            all = true;
        }
        if all {
            active = schema
                .columns()
                .iter()
                .filter(|column| column.semantics.is_some())
                .map(|column| column.name.clone())
                .collect();
            all = false;
        }
        Self { all, active }
    }

    fn contains(&self, column_name: &str) -> bool {
        self.all || self.active.contains(column_name)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlannerMetadataCost {
    pub selectivity: f64,
    pub supported_qual_count: usize,
}

#[derive(Default)]
struct QualDiagnostics {
    supported_qual_count: usize,
    unsupported_qual_count: usize,
    top_level_qual_count: usize,
    plan_qual_count: usize,
    index_qual_count: usize,
    detected_columns: BTreeSet<String>,
    detected_values: BTreeSet<String>,
    detected_param_values: BTreeSet<String>,
    predicates: Vec<MetadataPredicate>,
    unavailable_param_count: usize,
}

impl QualDiagnostics {
    fn record_supported(&mut self, predicate: MetadataPredicate, description: String) {
        self.supported_qual_count += 1;
        self.detected_columns.insert(predicate.column_name.clone());
        self.detected_values.insert(description);
        self.predicates.push(predicate);
    }

    fn record_unsupported(&mut self) {
        self.unsupported_qual_count += 1;
    }

    fn all_quals_covered(&self) -> bool {
        self.top_level_qual_count > 0
            && self.supported_qual_count > 0
            && self.unsupported_qual_count == 0
    }

    fn compiled(self) -> CompiledMetadataQual {
        let all_quals_covered = self.all_quals_covered();
        CompiledMetadataQual {
            predicates: self.predicates,
            supported_qual_count: self.supported_qual_count,
            unsupported_qual_count: self.unsupported_qual_count,
            all_quals_covered,
            unavailable_param_count: self.unavailable_param_count,
        }
    }
}

pub unsafe fn compile_scan_qual(
    scan: pg_sys::IndexScanDesc,
    hack: Option<NonNull<pg_sys::IndexScanState>>,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> CompiledMetadataQual {
    let Some(diagnostics) = (unsafe { inspect_scan_quals(scan, hack, schema, active) }) else {
        return CompiledMetadataQual::default();
    };
    diagnostics.compiled()
}

pub unsafe fn planner_metadata_cost(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> Option<PlannerMetadataCost> {
    unsafe {
        if root.is_null() || path.is_null() || schema.cols() == 0 {
            return None;
        }
        let index_info = (*path).indexinfo;
        if index_info.is_null() {
            return None;
        }
        let qual_list = (*index_info).indrestrictinfo;
        let mut metadata_quals = ptr::null_mut();
        let mut supported_qual_count = 0_usize;
        for_each_list_node(qual_list, |node| {
            if node.is_null() || (*node).type_ != pg_sys::NodeTag::T_RestrictInfo {
                return;
            }
            let rinfo = node.cast::<pg_sys::RestrictInfo>();
            let clause = (*rinfo).clause.cast::<pg_sys::Node>();
            if planner_qual_is_coverable(clause, index_info, schema, active) {
                supported_qual_count += 1;
                metadata_quals = pg_sys::lappend(metadata_quals, rinfo.cast());
            }
        });
        if supported_qual_count == 0 || metadata_quals.is_null() {
            return None;
        }
        let var_relid = if (*index_info).rel.is_null() {
            0
        } else {
            (*(*index_info).rel).relid as i32
        };
        let selectivity = pg_sys::clauselist_selectivity(
            root,
            metadata_quals,
            var_relid,
            pg_sys::JoinType::JOIN_INNER,
            ptr::null_mut(),
        );
        Some(PlannerMetadataCost {
            selectivity,
            supported_qual_count,
        })
    }
}

pub unsafe fn log_scan_qual_diagnostics(
    scan: pg_sys::IndexScanDesc,
    hack: Option<NonNull<pg_sys::IndexScanState>>,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) {
    let Some(hack) = hack else {
        pgrx::log!(
            "vchordrq_metadata_qual_diagnostics hack_present=false metadata_supported_qual_count=0 metadata_unsupported_qual_count=0 metadata_detected_columns=none metadata_detected_param_values=none metadata_detected_values=none metadata_all_quals_covered=false"
        );
        return;
    };
    unsafe {
        let scan_state = hack.as_ptr();
        let plan_state = &raw const (*scan_state).ss.ps;
        let plan = (*plan_state).plan;
        if plan.is_null() {
            pgrx::log!(
                "vchordrq_metadata_qual_diagnostics hack_present=true plan_present=false metadata_supported_qual_count=0 metadata_unsupported_qual_count=0 metadata_detected_columns=none metadata_detected_param_values=none metadata_detected_values=none metadata_all_quals_covered=false"
            );
            return;
        }
        let plan_tag = (*plan).type_;
        let diagnostics = inspect_scan_quals(scan, Some(hack), schema, active).unwrap_or_default();
        let external_param_count = external_param_count(plan_state.cast_mut());
        pgrx::log!(
            "vchordrq_metadata_qual_diagnostics hack_present=true plan_present=true plan_tag={:?} metadata_supported_qual_count={} metadata_unsupported_qual_count={} metadata_detected_columns={} metadata_detected_param_values={} metadata_detected_values={} metadata_all_quals_covered={} metadata_unavailable_param_count={} top_level_qual_count={} plan_qual_count={} index_qual_count={} runtime_keys_ready={} runtime_key_count={} external_param_count={}",
            plan_tag,
            diagnostics.supported_qual_count,
            diagnostics.unsupported_qual_count,
            join_set(&diagnostics.detected_columns),
            join_set(&diagnostics.detected_param_values),
            join_set(&diagnostics.detected_values),
            diagnostics.all_quals_covered(),
            diagnostics.unavailable_param_count,
            diagnostics.top_level_qual_count,
            diagnostics.plan_qual_count,
            diagnostics.index_qual_count,
            (*scan_state).iss_RuntimeKeysReady,
            (*scan_state).iss_NumRuntimeKeys,
            external_param_count,
        );
    }
}

unsafe fn inspect_scan_quals(
    scan: pg_sys::IndexScanDesc,
    hack: Option<NonNull<pg_sys::IndexScanState>>,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> Option<QualDiagnostics> {
    unsafe {
        let hack = hack?;
        let scan_state = hack.as_ptr();
        let plan_state = &raw const (*scan_state).ss.ps;
        let plan = (*plan_state).plan;
        if plan.is_null() {
            return None;
        }
        let heap_relation = (*scan).heapRelation;
        let mut diagnostics = QualDiagnostics::default();
        inspect_qual_list(
            (*plan).qual,
            heap_relation,
            plan_state.cast_mut(),
            schema,
            active,
            &mut diagnostics,
        );
        diagnostics.plan_qual_count = diagnostics.top_level_qual_count;
        if (*plan).type_ == pg_sys::NodeTag::T_IndexScan {
            let index_plan = plan.cast::<pg_sys::IndexScan>();
            let before = diagnostics.top_level_qual_count;
            inspect_qual_list(
                (*index_plan).indexqualorig,
                heap_relation,
                plan_state.cast_mut(),
                schema,
                active,
                &mut diagnostics,
            );
            diagnostics.index_qual_count = diagnostics.top_level_qual_count - before;
        }
        Some(diagnostics)
    }
}

unsafe fn inspect_qual_list(
    list: *mut pg_sys::List,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
    diagnostics: &mut QualDiagnostics,
) {
    if list.is_null() {
        return;
    }
    unsafe {
        let len = (*list).length.max(0) as usize;
        let cells = (*list).elements;
        for i in 0..len {
            let node = (*cells.add(i)).ptr_value.cast::<pg_sys::Node>();
            if node.is_null() {
                continue;
            }
            diagnostics.top_level_qual_count += 1;
            inspect_qual(node, heap_relation, plan_state, schema, active, diagnostics);
        }
    }
}

unsafe fn inspect_qual(
    node: *mut pg_sys::Node,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
    diagnostics: &mut QualDiagnostics,
) -> bool {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() {
            diagnostics.record_unsupported();
            return false;
        }
        match (*node).type_ {
            pg_sys::NodeTag::T_BoolExpr => {
                let expr = node.cast::<pg_sys::BoolExpr>();
                if (*expr).boolop != pg_sys::BoolExprType::AND_EXPR {
                    collect_detected_columns(node, heap_relation, schema, diagnostics);
                    diagnostics.record_unsupported();
                    return false;
                }
                let mut all_supported = true;
                for_each_list_node((*expr).args, |child| {
                    all_supported &= inspect_qual(
                        child,
                        heap_relation,
                        plan_state,
                        schema,
                        active,
                        diagnostics,
                    );
                });
                all_supported
            }
            pg_sys::NodeTag::T_OpExpr
                if inspect_op_expr(
                    node.cast(),
                    heap_relation,
                    plan_state,
                    schema,
                    active,
                    diagnostics,
                ) =>
            {
                true
            }
            pg_sys::NodeTag::T_ScalarArrayOpExpr
                if inspect_scalar_array_expr(
                    node.cast(),
                    heap_relation,
                    plan_state,
                    schema,
                    active,
                    diagnostics,
                ) =>
            {
                true
            }
            _ => {
                collect_detected_columns(node, heap_relation, schema, diagnostics);
                diagnostics.record_unsupported();
                false
            }
        }
    }
}

unsafe fn inspect_op_expr(
    expr: *mut pg_sys::OpExpr,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
    diagnostics: &mut QualDiagnostics,
) -> bool {
    unsafe {
        let args = list_nodes((*expr).args);
        if args.len() != 2 {
            return false;
        }
        if let Some(detected) =
            inspect_bitmask_contains(expr, args[0], args[1], heap_relation, plan_state, schema)
        {
            return record_detected(detected, active, diagnostics);
        }
        if let Some(detected) = inspect_binary_predicate(
            expr,
            args[0],
            args[1],
            heap_relation,
            plan_state,
            schema,
            diagnostics,
        ) {
            return record_detected(detected, active, diagnostics);
        }
        if let Some(detected) = inspect_binary_predicate(
            expr,
            args[1],
            args[0],
            heap_relation,
            plan_state,
            schema,
            diagnostics,
        ) {
            return record_detected(detected.invert(), active, diagnostics);
        }
        false
    }
}

unsafe fn inspect_scalar_array_expr(
    expr: *mut pg_sys::ScalarArrayOpExpr,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
    diagnostics: &mut QualDiagnostics,
) -> bool {
    unsafe {
        if !(*expr).useOr || !is_int8_equality_func((*expr).opfuncid) {
            return false;
        }
        let args = list_nodes((*expr).args);
        if args.len() != 2 {
            return false;
        }
        let Some(var) = metadata_var(args[0], heap_relation, schema) else {
            return false;
        };
        if var.vartype != pg_sys::INT8OID {
            return false;
        }
        if !var.semantics.supports(MetadataColumnOp::In) {
            return false;
        }
        let Some(values) = array_values_for_node(args[1], plan_state, &var.name) else {
            return false;
        };
        if values.unavailable_param {
            diagnostics.unavailable_param_count += 1;
        }
        for param in values.param_descriptions {
            diagnostics.detected_param_values.insert(param);
        }
        if !values.supported {
            return false;
        }
        record_detected(
            DetectedPredicate {
                predicate: MetadataPredicate {
                    metadata_index: var.metadata_index,
                    column_name: var.name.clone(),
                    exact: var.semantics.exact,
                    op: MetadataPredicateOp::In(values.values),
                },
                description: format!("{}=ANY({})", var.name, values.description),
            },
            active,
            diagnostics,
        )
    }
}

fn record_detected(
    detected: DetectedPredicate,
    active: &MetadataActiveColumns,
    diagnostics: &mut QualDiagnostics,
) -> bool {
    if !active.contains(&detected.predicate.column_name) {
        return false;
    }
    diagnostics.record_supported(detected.predicate, detected.description);
    true
}

unsafe fn planner_qual_is_coverable(
    node: *mut pg_sys::Node,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> bool {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() {
            return false;
        }
        match (*node).type_ {
            pg_sys::NodeTag::T_BoolExpr => {
                let expr = node.cast::<pg_sys::BoolExpr>();
                if (*expr).boolop != pg_sys::BoolExprType::AND_EXPR {
                    return false;
                }
                let args = list_nodes((*expr).args);
                !args.is_empty()
                    && args
                        .into_iter()
                        .all(|child| planner_qual_is_coverable(child, index_info, schema, active))
            }
            pg_sys::NodeTag::T_OpExpr => {
                planner_op_expr_is_coverable(node.cast(), index_info, schema, active)
            }
            pg_sys::NodeTag::T_ScalarArrayOpExpr => {
                planner_scalar_array_expr_is_coverable(node.cast(), index_info, schema, active)
            }
            _ => false,
        }
    }
}

unsafe fn planner_op_expr_is_coverable(
    expr: *mut pg_sys::OpExpr,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> bool {
    unsafe {
        let args = list_nodes((*expr).args);
        if args.len() != 2 {
            return false;
        }
        planner_bitmask_contains_is_coverable(expr, args[0], args[1], index_info, schema, active)
            || planner_binary_predicate_is_coverable(
                expr, args[0], args[1], index_info, schema, active,
            )
            || planner_binary_predicate_is_coverable(
                expr, args[1], args[0], index_info, schema, active,
            )
    }
}

unsafe fn planner_scalar_array_expr_is_coverable(
    expr: *mut pg_sys::ScalarArrayOpExpr,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> bool {
    unsafe {
        if !(*expr).useOr || !is_int8_equality_func((*expr).opfuncid) {
            return false;
        }
        let args = list_nodes((*expr).args);
        if args.len() != 2 {
            return false;
        }
        let Some(var) = planner_metadata_var(args[0], index_info, schema) else {
            return false;
        };
        if var.vartype != pg_sys::INT8OID
            || !active.contains(&var.name)
            || !var.semantics.supports(MetadataColumnOp::In)
        {
            return false;
        }
        array_values_for_node(args[1], ptr::null_mut(), &var.name)
            .map(|values| values.supported)
            .unwrap_or(false)
    }
}

unsafe fn planner_binary_predicate_is_coverable(
    expr: *mut pg_sys::OpExpr,
    var_node: *mut pg_sys::Node,
    value_node: *mut pg_sys::Node,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> bool {
    unsafe {
        let Some(var) = planner_metadata_var(var_node, index_info, schema) else {
            return false;
        };
        let Some(op) = int8_binary_op(expr) else {
            return false;
        };
        if var.vartype != pg_sys::INT8OID
            || !active.contains(&var.name)
            || !var.semantics.supports(metadata_op_for_binary_op(op))
        {
            return false;
        }
        scalar_value_for_node(value_node, ptr::null_mut(), &var.name)
            .map(|value| value.supported)
            .unwrap_or(false)
    }
}

unsafe fn planner_bitmask_contains_is_coverable(
    expr: *mut pg_sys::OpExpr,
    left: *mut pg_sys::Node,
    right: *mut pg_sys::Node,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
    active: &MetadataActiveColumns,
) -> bool {
    unsafe {
        if int8_binary_op(expr) != Some(Int8BinaryOp::Eq) {
            return false;
        }
        let Some((var, mask)) = planner_int8_and(left, index_info, schema) else {
            return false;
        };
        if !active.contains(&var.name) || !var.semantics.supports(MetadataColumnOp::BitmaskContains)
        {
            return false;
        }
        let Some(expected) = scalar_value_for_node(right, ptr::null_mut(), &var.name) else {
            return false;
        };
        expected.supported && expected.value == Some(mask)
    }
}

unsafe fn planner_int8_and(
    node: *mut pg_sys::Node,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
) -> Option<(MetadataVar, i64)> {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() || (*node).type_ != pg_sys::NodeTag::T_OpExpr {
            return None;
        }
        let expr = node.cast::<pg_sys::OpExpr>();
        if !is_int8_and_operator(expr) {
            return None;
        }
        let args = list_nodes((*expr).args);
        if args.len() != 2 {
            return None;
        }
        planner_int8_and_operands(args[0], args[1], index_info, schema)
            .or_else(|| planner_int8_and_operands(args[1], args[0], index_info, schema))
    }
}

unsafe fn planner_int8_and_operands(
    var_node: *mut pg_sys::Node,
    value_node: *mut pg_sys::Node,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
) -> Option<(MetadataVar, i64)> {
    let var = unsafe { planner_metadata_var(var_node, index_info, schema)? };
    if var.vartype != pg_sys::INT8OID {
        return None;
    }
    let mask = unsafe { scalar_value_for_node(value_node, ptr::null_mut(), &var.name)? };
    if mask.supported {
        Some((var, mask.value?))
    } else {
        None
    }
}

#[derive(Debug)]
struct DetectedPredicate {
    predicate: MetadataPredicate,
    description: String,
}

impl DetectedPredicate {
    fn invert(mut self) -> Self {
        self.predicate.op = match self.predicate.op {
            MetadataPredicateOp::Eq(value) => MetadataPredicateOp::Eq(value),
            MetadataPredicateOp::Ge(value) => MetadataPredicateOp::Le(value),
            MetadataPredicateOp::Gt(value) => MetadataPredicateOp::Lt(value),
            MetadataPredicateOp::Le(value) => MetadataPredicateOp::Ge(value),
            MetadataPredicateOp::Lt(value) => MetadataPredicateOp::Gt(value),
            MetadataPredicateOp::In(values) => MetadataPredicateOp::In(values),
            MetadataPredicateOp::BitmaskContains(mask) => {
                MetadataPredicateOp::BitmaskContains(mask)
            }
        };
        self
    }
}

unsafe fn inspect_binary_predicate(
    expr: *mut pg_sys::OpExpr,
    var_node: *mut pg_sys::Node,
    value_node: *mut pg_sys::Node,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
    diagnostics: &mut QualDiagnostics,
) -> Option<DetectedPredicate> {
    unsafe {
        let var = metadata_var(var_node, heap_relation, schema)?;
        if var.vartype != pg_sys::INT8OID {
            return None;
        }
        let op = int8_binary_op(expr)?;
        if !var.semantics.supports(metadata_op_for_binary_op(op)) {
            return None;
        }
        let value = scalar_value_for_node(value_node, plan_state, &var.name)?;
        if value.unavailable_param {
            diagnostics.unavailable_param_count += 1;
        }
        if let Some(param) = &value.param_description {
            diagnostics.detected_param_values.insert(param.clone());
        }
        if !value.supported {
            return None;
        }
        let target = value.value?;
        Some(DetectedPredicate {
            predicate: MetadataPredicate {
                metadata_index: var.metadata_index,
                column_name: var.name.clone(),
                exact: var.semantics.exact,
                op: match op {
                    Int8BinaryOp::Eq => MetadataPredicateOp::Eq(target),
                    Int8BinaryOp::Ge => MetadataPredicateOp::Ge(target),
                    Int8BinaryOp::Gt => MetadataPredicateOp::Gt(target),
                    Int8BinaryOp::Le => MetadataPredicateOp::Le(target),
                    Int8BinaryOp::Lt => MetadataPredicateOp::Lt(target),
                },
            },
            description: format!("{}{}{}", var.name, op.symbol(), value.description),
        })
    }
}

unsafe fn inspect_bitmask_contains(
    expr: *mut pg_sys::OpExpr,
    left: *mut pg_sys::Node,
    right: *mut pg_sys::Node,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
) -> Option<DetectedPredicate> {
    unsafe {
        if int8_binary_op(expr)? != Int8BinaryOp::Eq {
            return None;
        }
        let (var, mask) = inspect_int8_and(left, heap_relation, plan_state, schema)?;
        if !var.semantics.supports(MetadataColumnOp::BitmaskContains) {
            return None;
        }
        let expected = scalar_value_for_node(right, plan_state, &var.name)?;
        if !expected.supported || expected.value? != mask {
            return None;
        }
        Some(DetectedPredicate {
            predicate: MetadataPredicate {
                metadata_index: var.metadata_index,
                column_name: var.name.clone(),
                exact: var.semantics.exact,
                op: MetadataPredicateOp::BitmaskContains(mask),
            },
            description: format!("({}&{})={}", var.name, mask, mask),
        })
    }
}

unsafe fn inspect_int8_and(
    node: *mut pg_sys::Node,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
) -> Option<(MetadataVar, i64)> {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() || (*node).type_ != pg_sys::NodeTag::T_OpExpr {
            return None;
        }
        let expr = node.cast::<pg_sys::OpExpr>();
        if !is_int8_and_operator(expr) {
            return None;
        }
        let args = list_nodes((*expr).args);
        if args.len() != 2 {
            return None;
        }
        if let Some(result) =
            inspect_int8_and_operands(args[0], args[1], heap_relation, plan_state, schema)
        {
            return Some(result);
        }
        inspect_int8_and_operands(args[1], args[0], heap_relation, plan_state, schema)
    }
}

unsafe fn inspect_int8_and_operands(
    var_node: *mut pg_sys::Node,
    value_node: *mut pg_sys::Node,
    heap_relation: pg_sys::Relation,
    plan_state: *mut pg_sys::PlanState,
    schema: &MetadataSchema,
) -> Option<(MetadataVar, i64)> {
    let var = unsafe { metadata_var(var_node, heap_relation, schema)? };
    if var.vartype != pg_sys::INT8OID {
        return None;
    }
    let mask = unsafe { scalar_value_for_node(value_node, plan_state, &var.name)? };
    if mask.supported {
        Some((var, mask.value?))
    } else {
        None
    }
}

#[derive(Debug)]
struct DetectedScalar {
    supported: bool,
    value: Option<i64>,
    unavailable_param: bool,
    description: String,
    param_description: Option<String>,
}

#[derive(Debug)]
struct DetectedArray {
    supported: bool,
    values: Vec<i64>,
    unavailable_param: bool,
    description: String,
    param_descriptions: Vec<String>,
}

unsafe fn scalar_value_for_node(
    node: *mut pg_sys::Node,
    plan_state: *mut pg_sys::PlanState,
    column_name: &str,
) -> Option<DetectedScalar> {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() {
            return None;
        }
        match (*node).type_ {
            pg_sys::NodeTag::T_Const => {
                let value = node.cast::<pg_sys::Const>();
                if !is_supported_integer_type((*value).consttype) {
                    return Some(DetectedScalar {
                        supported: false,
                        value: None,
                        unavailable_param: false,
                        description: format!(
                            "const(<unsupported-type:{}>)",
                            (*value).consttype.to_u32()
                        ),
                        param_description: None,
                    });
                }
                if (*value).constisnull {
                    return Some(DetectedScalar {
                        supported: false,
                        value: None,
                        unavailable_param: false,
                        description: "const(NULL)".to_owned(),
                        param_description: None,
                    });
                }
                let value = integer_datum_value((*value).consttype, (*value).constvalue);
                Some(DetectedScalar {
                    supported: true,
                    value: Some(value),
                    unavailable_param: false,
                    description: format!("const({value})"),
                    param_description: None,
                })
            }
            pg_sys::NodeTag::T_Param => {
                let param = node.cast::<pg_sys::Param>();
                if !is_supported_integer_type((*param).paramtype) {
                    return Some(DetectedScalar {
                        supported: false,
                        value: None,
                        unavailable_param: false,
                        description: format!(
                            "param${}(<unsupported-type:{}>)",
                            (*param).paramid,
                            (*param).paramtype.to_u32()
                        ),
                        param_description: Some(format!(
                            "${}=<unsupported-type:{}>",
                            (*param).paramid,
                            (*param).paramtype.to_u32()
                        )),
                    });
                }
                let fetched = fetch_param_value(plan_state, &*param);
                let (supported, value, unavailable_param, param_value) = match fetched {
                    ParamValue::Int8(value) => (true, Some(value), false, value.to_string()),
                    ParamValue::Null => (false, None, false, "NULL".to_owned()),
                    ParamValue::Unavailable => (false, None, true, "unavailable".to_owned()),
                    ParamValue::UnsupportedKind(kind) => {
                        (false, None, false, format!("unsupported-kind:{kind}"))
                    }
                    ParamValue::Int8Array(_) => (false, None, false, "array".to_owned()),
                };
                Some(DetectedScalar {
                    supported,
                    value,
                    unavailable_param,
                    description: format!("param${}({})", (*param).paramid, param_value),
                    param_description: Some(format!("${}={}", (*param).paramid, param_value)),
                })
            }
            _ => {
                let _ = column_name;
                None
            }
        }
    }
}

unsafe fn array_values_for_node(
    node: *mut pg_sys::Node,
    plan_state: *mut pg_sys::PlanState,
    column_name: &str,
) -> Option<DetectedArray> {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() {
            return None;
        }
        match (*node).type_ {
            pg_sys::NodeTag::T_Const => {
                let value = node.cast::<pg_sys::Const>();
                if (*value).consttype != pg_sys::INT8ARRAYOID || (*value).constisnull {
                    return Some(DetectedArray {
                        supported: false,
                        values: Vec::new(),
                        unavailable_param: false,
                        description: "const(<unsupported-array>)".to_owned(),
                        param_descriptions: Vec::new(),
                    });
                }
                let values = int8_array_values((*value).constvalue);
                Some(DetectedArray {
                    supported: true,
                    description: format!("const_array({})", values.len()),
                    values,
                    unavailable_param: false,
                    param_descriptions: Vec::new(),
                })
            }
            pg_sys::NodeTag::T_Param => {
                let param = node.cast::<pg_sys::Param>();
                if (*param).paramtype != pg_sys::INT8ARRAYOID {
                    return Some(DetectedArray {
                        supported: false,
                        values: Vec::new(),
                        unavailable_param: false,
                        description: format!(
                            "param${}(<unsupported-type:{}>)",
                            (*param).paramid,
                            (*param).paramtype.to_u32()
                        ),
                        param_descriptions: vec![format!(
                            "${}=<unsupported-type:{}>",
                            (*param).paramid,
                            (*param).paramtype.to_u32()
                        )],
                    });
                }
                let fetched = fetch_param_value(plan_state, &*param);
                let (supported, values, unavailable_param, param_value) = match fetched {
                    ParamValue::Int8Array(values) => {
                        let len = values.len();
                        (true, values, false, format!("array({len})"))
                    }
                    ParamValue::Null => (false, Vec::new(), false, "NULL".to_owned()),
                    ParamValue::Unavailable => (false, Vec::new(), true, "unavailable".to_owned()),
                    ParamValue::UnsupportedKind(kind) => {
                        (false, Vec::new(), false, format!("unsupported-kind:{kind}"))
                    }
                    ParamValue::Int8(value) => (false, vec![value], false, "scalar".to_owned()),
                };
                Some(DetectedArray {
                    supported,
                    values,
                    unavailable_param,
                    description: format!("param${}({})", (*param).paramid, param_value),
                    param_descriptions: vec![format!("${}={}", (*param).paramid, param_value)],
                })
            }
            _ => {
                let _ = column_name;
                None
            }
        }
    }
}

enum ParamValue {
    Int8(i64),
    Int8Array(Vec<i64>),
    Null,
    Unavailable,
    UnsupportedKind(pg_sys::ParamKind::Type),
}

unsafe fn fetch_param_value(
    plan_state: *mut pg_sys::PlanState,
    param: &pg_sys::Param,
) -> ParamValue {
    unsafe {
        if param.paramkind != pg_sys::ParamKind::PARAM_EXTERN {
            return ParamValue::UnsupportedKind(param.paramkind);
        }
        if param.paramid <= 0 || plan_state.is_null() || (*plan_state).state.is_null() {
            return ParamValue::Unavailable;
        }
        let params = (*(*plan_state).state).es_param_list_info;
        if params.is_null() || param.paramid > (*params).numParams {
            return ParamValue::Unavailable;
        }
        let param_data = if (*params).paramFetch.is_some() {
            return ParamValue::Unavailable;
        } else {
            (*params)
                .params
                .as_ptr()
                .add((param.paramid - 1) as usize)
                .cast_mut()
        };
        if param_data.is_null() {
            return ParamValue::Unavailable;
        }
        if (*param_data).isnull {
            return ParamValue::Null;
        }
        match (*param_data).ptype {
            pg_sys::INT8OID | pg_sys::INT4OID | pg_sys::INT2OID => ParamValue::Int8(
                integer_datum_value((*param_data).ptype, (*param_data).value),
            ),
            pg_sys::INT8ARRAYOID => ParamValue::Int8Array(int8_array_values((*param_data).value)),
            _ => ParamValue::Unavailable,
        }
    }
}

unsafe fn int8_array_values(datum: pg_sys::Datum) -> Vec<i64> {
    unsafe {
        let mut typlen = 0;
        let mut typbyval = false;
        let mut typalign = 0;
        pg_sys::get_typlenbyvalalign(pg_sys::INT8OID, &mut typlen, &mut typbyval, &mut typalign);
        let mut elements: *mut pg_sys::Datum = ptr::null_mut();
        let mut nulls: *mut bool = ptr::null_mut();
        let mut nelems = 0;
        pg_sys::deconstruct_array(
            datum.cast_mut_ptr::<pg_sys::ArrayType>(),
            pg_sys::INT8OID,
            typlen.into(),
            typbyval,
            typalign,
            &mut elements,
            &mut nulls,
            &mut nelems,
        );
        let mut values = Vec::with_capacity(nelems.max(0) as usize);
        for i in 0..nelems.max(0) as usize {
            if !nulls.is_null() && nulls.add(i).read() {
                continue;
            }
            values.push(elements.add(i).read().value() as i64);
        }
        values
    }
}

unsafe fn external_param_count(plan_state: *mut pg_sys::PlanState) -> i32 {
    unsafe {
        if plan_state.is_null() || (*plan_state).state.is_null() {
            return 0;
        }
        let params = (*(*plan_state).state).es_param_list_info;
        if params.is_null() {
            0
        } else {
            (*params).numParams
        }
    }
}

#[derive(Debug, Clone)]
struct MetadataVar {
    name: String,
    semantics: MetadataColumnSemantics,
    metadata_index: usize,
    vartype: pg_sys::Oid,
}

unsafe fn metadata_var(
    node: *mut pg_sys::Node,
    heap_relation: pg_sys::Relation,
    schema: &MetadataSchema,
) -> Option<MetadataVar> {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() || (*node).type_ != pg_sys::NodeTag::T_Var {
            return None;
        }
        let var = *node.cast::<pg_sys::Var>();
        let name = attname(heap_relation, var.varattno)
            .or_else(|| attname(heap_relation, var.varattnosyn))?;
        let column = schema.by_name(&name)?;
        let semantics = column.semantics.clone()?;
        Some(MetadataVar {
            name,
            semantics,
            metadata_index: column.metadata_index,
            vartype: var.vartype,
        })
    }
}

unsafe fn planner_metadata_var(
    node: *mut pg_sys::Node,
    index_info: *mut pg_sys::IndexOptInfo,
    schema: &MetadataSchema,
) -> Option<MetadataVar> {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null()
            || index_info.is_null()
            || (*index_info).indexkeys.is_null()
            || (*node).type_ != pg_sys::NodeTag::T_Var
        {
            return None;
        }
        let var = *node.cast::<pg_sys::Var>();
        if !(*index_info).rel.is_null() && var.varno != (*(*index_info).rel).relid as i32 {
            return None;
        }
        for column in schema.columns() {
            if column.index_attno >= (*index_info).ncolumns as usize {
                continue;
            }
            let heap_attno = (*index_info).indexkeys.add(column.index_attno).read();
            if heap_attno <= 0 {
                continue;
            }
            if var.varattno as i32 == heap_attno || var.varattnosyn as i32 == heap_attno {
                let semantics = column.semantics.clone()?;
                return Some(MetadataVar {
                    name: column.name.clone(),
                    semantics,
                    metadata_index: column.metadata_index,
                    vartype: var.vartype,
                });
            }
        }
        None
    }
}

unsafe fn attname(heap_relation: pg_sys::Relation, attno: pg_sys::AttrNumber) -> Option<String> {
    unsafe {
        if heap_relation.is_null() || attno <= 0 {
            return None;
        }
        let relid = (*heap_relation).rd_id;
        let name = pg_sys::get_attname(relid, attno, true);
        if name.is_null() {
            return None;
        }
        let result = CStr::from_ptr(name).to_str().ok().map(str::to_owned);
        pg_sys::pfree(name.cast());
        result
    }
}

unsafe fn collect_detected_columns(
    node: *mut pg_sys::Node,
    heap_relation: pg_sys::Relation,
    schema: &MetadataSchema,
    diagnostics: &mut QualDiagnostics,
) {
    unsafe {
        let node = strip_relabel(node);
        if node.is_null() {
            return;
        }
        match (*node).type_ {
            pg_sys::NodeTag::T_Var => {
                if let Some(var) = metadata_var(node, heap_relation, schema) {
                    diagnostics.detected_columns.insert(var.name);
                }
            }
            pg_sys::NodeTag::T_OpExpr => {
                let expr = node.cast::<pg_sys::OpExpr>();
                for_each_list_node((*expr).args, |child| {
                    collect_detected_columns(child, heap_relation, schema, diagnostics);
                });
            }
            pg_sys::NodeTag::T_BoolExpr => {
                let expr = node.cast::<pg_sys::BoolExpr>();
                for_each_list_node((*expr).args, |child| {
                    collect_detected_columns(child, heap_relation, schema, diagnostics);
                });
            }
            pg_sys::NodeTag::T_ScalarArrayOpExpr => {
                let expr = node.cast::<pg_sys::ScalarArrayOpExpr>();
                for_each_list_node((*expr).args, |child| {
                    collect_detected_columns(child, heap_relation, schema, diagnostics);
                });
            }
            pg_sys::NodeTag::T_NullTest => {
                let expr = node.cast::<pg_sys::NullTest>();
                collect_detected_columns((*expr).arg.cast(), heap_relation, schema, diagnostics);
            }
            _ => {}
        }
    }
}

unsafe fn strip_relabel(mut node: *mut pg_sys::Node) -> *mut pg_sys::Node {
    unsafe {
        loop {
            if node.is_null() || (*node).type_ != pg_sys::NodeTag::T_RelabelType {
                return node;
            }
            node = (*node.cast::<pg_sys::RelabelType>()).arg.cast();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Int8BinaryOp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
}

impl Int8BinaryOp {
    const fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ge => ">=",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Lt => "<",
        }
    }
}

fn metadata_op_for_binary_op(op: Int8BinaryOp) -> MetadataColumnOp {
    match op {
        Int8BinaryOp::Eq => MetadataColumnOp::Eq,
        Int8BinaryOp::Ge | Int8BinaryOp::Gt | Int8BinaryOp::Le | Int8BinaryOp::Lt => {
            MetadataColumnOp::Range
        }
    }
}

unsafe fn int8_binary_op(expr: *mut pg_sys::OpExpr) -> Option<Int8BinaryOp> {
    unsafe {
        let func = opfunc_id(expr).to_u32();
        match func {
            pg_sys::F_INT8EQ
            | pg_sys::F_INT84EQ
            | pg_sys::F_INT82EQ
            | pg_sys::F_INT48EQ
            | pg_sys::F_INT28EQ => Some(Int8BinaryOp::Eq),
            pg_sys::F_INT8GE
            | pg_sys::F_INT84GE
            | pg_sys::F_INT82GE
            | pg_sys::F_INT48GE
            | pg_sys::F_INT28GE => Some(Int8BinaryOp::Ge),
            pg_sys::F_INT8GT
            | pg_sys::F_INT84GT
            | pg_sys::F_INT82GT
            | pg_sys::F_INT48GT
            | pg_sys::F_INT28GT => Some(Int8BinaryOp::Gt),
            pg_sys::F_INT8LE
            | pg_sys::F_INT84LE
            | pg_sys::F_INT82LE
            | pg_sys::F_INT48LE
            | pg_sys::F_INT28LE => Some(Int8BinaryOp::Le),
            pg_sys::F_INT8LT
            | pg_sys::F_INT84LT
            | pg_sys::F_INT82LT
            | pg_sys::F_INT48LT
            | pg_sys::F_INT28LT => Some(Int8BinaryOp::Lt),
            _ => None,
        }
    }
}

fn is_supported_integer_type(oid: pg_sys::Oid) -> bool {
    oid == pg_sys::INT8OID || oid == pg_sys::INT4OID || oid == pg_sys::INT2OID
}

fn integer_datum_value(oid: pg_sys::Oid, datum: pg_sys::Datum) -> i64 {
    match oid {
        pg_sys::INT8OID => datum.value() as i64,
        pg_sys::INT4OID => datum.value() as i32 as i64,
        pg_sys::INT2OID => datum.value() as i16 as i64,
        _ => datum.value() as i64,
    }
}

unsafe fn is_int8_equality_func(func: pg_sys::Oid) -> bool {
    func.to_u32() == pg_sys::F_INT8EQ
}

unsafe fn is_int8_and_operator(expr: *mut pg_sys::OpExpr) -> bool {
    unsafe { opfunc_id(expr).to_u32() == pg_sys::F_INT8AND }
}

unsafe fn opfunc_id(expr: *mut pg_sys::OpExpr) -> pg_sys::Oid {
    unsafe {
        if (*expr).opfuncid.to_u32() != 0 {
            (*expr).opfuncid
        } else {
            pg_sys::get_opcode((*expr).opno)
        }
    }
}

unsafe fn list_nodes(list: *mut pg_sys::List) -> Vec<*mut pg_sys::Node> {
    let mut nodes = Vec::new();
    unsafe {
        for_each_list_node(list, |node| nodes.push(node));
    }
    nodes
}

unsafe fn for_each_list_node(list: *mut pg_sys::List, mut f: impl FnMut(*mut pg_sys::Node)) {
    if list.is_null() {
        return;
    }
    unsafe {
        let len = (*list).length.max(0) as usize;
        let cells = (*list).elements;
        for i in 0..len {
            let node = (*cells.add(i)).ptr_value.cast::<pg_sys::Node>();
            if !node.is_null() {
                f(node);
            }
        }
    }
}

fn join_set(set: &BTreeSet<String>) -> String {
    if set.is_empty() {
        "none".to_owned()
    } else {
        set.iter().cloned().collect::<Vec<_>>().join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::super::metadata::MetadataColumn;
    use super::*;

    fn metadata(values: &[(usize, i64)]) -> vchordrq::CandidateMetadata {
        let mut metadata = vchordrq::CandidateMetadata::default();
        for &(index, value) in values {
            metadata.set(index, value);
        }
        metadata
    }

    fn test_schema() -> MetadataSchema {
        MetadataSchema::new_for_test(vec![
            MetadataColumn {
                name: "tenant_hash".to_owned(),
                semantics: Some(MetadataColumnSemantics::new(
                    [MetadataColumnOp::Eq, MetadataColumnOp::In],
                    false,
                )),
                index_attno: 1,
                metadata_index: 0,
            },
            MetadataColumn {
                name: "state_code".to_owned(),
                semantics: Some(MetadataColumnSemantics::new(
                    [MetadataColumnOp::Eq, MetadataColumnOp::In],
                    true,
                )),
                index_attno: 2,
                metadata_index: 1,
            },
            MetadataColumn {
                name: "created_bucket".to_owned(),
                semantics: Some(MetadataColumnSemantics::new(
                    [MetadataColumnOp::Range],
                    false,
                )),
                index_attno: 3,
                metadata_index: 2,
            },
            MetadataColumn {
                name: "flags".to_owned(),
                semantics: Some(MetadataColumnSemantics::new(
                    [MetadataColumnOp::Eq, MetadataColumnOp::BitmaskContains],
                    true,
                )),
                index_attno: 4,
                metadata_index: 3,
            },
            MetadataColumn {
                name: "undeclared_include".to_owned(),
                semantics: None,
                index_attno: 5,
                metadata_index: 4,
            },
        ])
    }

    fn predicate(
        metadata_index: usize,
        column_name: &str,
        exact: bool,
        op: MetadataPredicateOp,
    ) -> MetadataPredicate {
        MetadataPredicate {
            metadata_index,
            column_name: column_name.to_owned(),
            exact,
            op,
        }
    }

    #[test]
    fn active_columns_parse_exact_names_all_and_skip_bogus() {
        let schema = test_schema();
        let active = MetadataActiveColumns::parse("tenant_hash, bogus, state_code", &schema);
        assert!(active.contains("tenant_hash"));
        assert!(active.contains("state_code"));
        assert!(!active.contains("flags"));
        assert!(!active.contains("bogus"));

        let all = MetadataActiveColumns::parse("", &schema);
        assert!(all.contains("tenant_hash"));
        assert!(all.contains("state_code"));
        assert!(all.contains("created_bucket"));
        assert!(all.contains("flags"));
        assert!(!all.contains("undeclared_include"));

        let retired_aliases = MetadataActiveColumns::parse("feed,geo,time", &schema);
        assert!(!retired_aliases.contains("tenant_hash"));
        assert!(!retired_aliases.contains("created_bucket"));
    }

    #[test]
    fn declared_semantics_gate_predicate_shapes() {
        let schema = test_schema();
        let tenant = schema
            .declared_by_name("tenant_hash")
            .and_then(|column| column.semantics.as_ref())
            .unwrap();
        assert!(tenant.supports(MetadataColumnOp::Eq));
        assert!(tenant.supports(MetadataColumnOp::In));
        assert!(!tenant.supports(MetadataColumnOp::Range));
        assert!(!tenant.supports(MetadataColumnOp::BitmaskContains));

        let created = schema
            .declared_by_name("created_bucket")
            .and_then(|column| column.semantics.as_ref())
            .unwrap();
        assert!(created.supports(MetadataColumnOp::Range));
        assert!(!created.supports(MetadataColumnOp::Eq));

        let flags = schema
            .declared_by_name("flags")
            .and_then(|column| column.semantics.as_ref())
            .unwrap();
        assert!(flags.supports(MetadataColumnOp::Eq));
        assert!(flags.supports(MetadataColumnOp::BitmaskContains));
        assert!(!flags.supports(MetadataColumnOp::Range));
    }

    #[test]
    fn predicate_shapes_reject_only_when_definitely_false() {
        let candidate = metadata(&[(0, 42), (1, 7), (2, 11), (3, 0b1011)]);

        assert_eq!(
            predicate(0, "tenant_hash", false, MetadataPredicateOp::Eq(42))
                .is_definitely_false(candidate),
            Some(false)
        );
        assert_eq!(
            predicate(0, "tenant_hash", false, MetadataPredicateOp::Eq(43))
                .is_definitely_false(candidate),
            Some(true)
        );
        assert_eq!(
            predicate(
                1,
                "state_code",
                true,
                MetadataPredicateOp::In(vec![3, 7, 9]),
            )
            .is_definitely_false(candidate),
            Some(false)
        );
        assert_eq!(
            predicate(2, "created_bucket", false, MetadataPredicateOp::Ge(12))
                .is_definitely_false(candidate),
            Some(true)
        );
        assert_eq!(
            predicate(2, "created_bucket", false, MetadataPredicateOp::Lt(12))
                .is_definitely_false(candidate),
            Some(false)
        );
        assert_eq!(
            predicate(
                3,
                "flags",
                true,
                MetadataPredicateOp::BitmaskContains(0b0011)
            )
            .is_definitely_false(candidate),
            Some(false)
        );
        assert_eq!(
            predicate(
                3,
                "flags",
                true,
                MetadataPredicateOp::BitmaskContains(0b0100)
            )
            .is_definitely_false(candidate),
            Some(true)
        );
    }

    #[test]
    fn missing_metadata_is_maybe_not_false() {
        let candidate = metadata(&[(0, 42)]);
        assert_eq!(
            predicate(1, "state_code", true, MetadataPredicateOp::Eq(9))
                .is_definitely_false(candidate),
            None
        );
    }

    #[test]
    fn exact_flag_controls_heap_recheck_skip() {
        assert!(
            predicate(1, "any_name", true, MetadataPredicateOp::Eq(1)).is_exact_for_heap_skip()
        );
        assert!(
            !predicate(1, "any_name", false, MetadataPredicateOp::Eq(1)).is_exact_for_heap_skip()
        );
    }

    #[test]
    fn diagnostics_counts_supported_unsupported_and_coverage() {
        let mut diagnostics = QualDiagnostics::default();
        diagnostics.top_level_qual_count = 2;
        diagnostics.record_supported(
            predicate(1, "state_code", true, MetadataPredicateOp::Eq(3)),
            "state_code=const(3)".to_owned(),
        );
        diagnostics.record_unsupported();
        diagnostics.unavailable_param_count = 1;

        let compiled = diagnostics.compiled();
        assert_eq!(compiled.supported_qual_count, 1);
        assert_eq!(compiled.unsupported_qual_count, 1);
        assert_eq!(compiled.unavailable_param_count, 1);
        assert!(!compiled.all_quals_covered);

        let mut covered = QualDiagnostics::default();
        covered.top_level_qual_count = 1;
        covered.record_supported(
            predicate(2, "created_bucket", false, MetadataPredicateOp::Ge(100)),
            "created_bucket>=const(100)".to_owned(),
        );
        assert!(covered.compiled().all_quals_covered);
    }
}
