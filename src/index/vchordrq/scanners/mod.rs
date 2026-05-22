// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

mod default;
mod maxsim;

use crate::index::gucs::MetadataPrefilterMode;
use crate::index::scanners::Io;
use crate::index::vchordrq::am::metadata::MetadataColumnKind;
use crate::index::vchordrq::am::metadata_qual::MetadataPredicate;
use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

pub use default::DefaultBuilder;
pub use maxsim::MaxsimBuilder;

#[derive(Debug, Clone)]
pub struct SearchInstrumentation {
    inner: Rc<SearchInstrumentationInner>,
}

#[derive(Debug)]
struct SearchInstrumentationInner {
    started_at: Instant,
    candidate_count: Cell<usize>,
    candidate_count_before_block_prune: Cell<usize>,
    candidate_count_after_block_prune: Cell<usize>,
    emitted_count: Cell<usize>,
    build_ns: Cell<u128>,
    rerank_next_ns: Cell<u128>,
    metadata_checked_count: Cell<usize>,
    metadata_rejected_count: Cell<usize>,
    metadata_survived_count: Cell<usize>,
    metadata_true_count: Cell<usize>,
    metadata_maybe_count: Cell<usize>,
    metadata_rejected_by_feed_count: Cell<usize>,
    metadata_rejected_by_flags_count: Cell<usize>,
    metadata_rejected_by_status_count: Cell<usize>,
    metadata_rejected_by_deleted_count: Cell<usize>,
    metadata_rejected_by_visibility_count: Cell<usize>,
    metadata_rejected_by_geo_count: Cell<usize>,
    metadata_rejected_by_time_count: Cell<usize>,
    residual_checked_count: Cell<usize>,
    residual_passed_count: Cell<usize>,
    residual_failed_count: Cell<usize>,
    hypothetical_reject_feed_count: Cell<usize>,
    hypothetical_reject_flags_count: Cell<usize>,
    hypothetical_reject_status_count: Cell<usize>,
    hypothetical_reject_deleted_count: Cell<usize>,
    hypothetical_reject_visibility_count: Cell<usize>,
    hypothetical_reject_geo_count: Cell<usize>,
    hypothetical_reject_time_count: Cell<usize>,
    hypothetical_reject_feed_flags_count: Cell<usize>,
    hypothetical_reject_feed_flags_geo_count: Cell<usize>,
    hypothetical_reject_feed_flags_geo_time_count: Cell<usize>,
    metadata_eval_ns: Cell<u128>,
    metadata_decode_ns: Cell<u128>,
    metadata_supported_qual_count: Cell<usize>,
    metadata_unsupported_qual_count: Cell<usize>,
    metadata_all_quals_covered: Cell<bool>,
    metadata_unavailable_param_count: Cell<usize>,
    heap_prefilter_after_metadata_count: Cell<usize>,
    heap_prefilter_avoided_count: Cell<usize>,
    metadata_false_negative_count_debug: Cell<usize>,
    block_summary_checked_count: Cell<usize>,
    block_summary_rejected_count: Cell<usize>,
    block_summary_maybe_count: Cell<usize>,
    block_summary_candidates_skipped_count: Cell<usize>,
    rerank: vchordrq::RerankInstrumentation,
    logged: Cell<bool>,
}

// finish_instrumentation in am/mod.rs only emits the metadata-relevant subset
// of these fields; the rest are intentionally kept populated for future
// instrumentation surfaces (streaming-IO / per-stage breakdown) without
// re-threading the snapshot wiring.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct SearchInstrumentationSnapshot {
    pub candidate_count: usize,
    pub candidate_count_before_block_prune: usize,
    pub candidate_count_after_block_prune: usize,
    pub scored_count: usize,
    pub emitted_count: usize,
    pub build_ns: u128,
    pub rerank_next_ns: u128,
    pub total_ns: u128,
    pub metadata_checked_count: usize,
    pub metadata_rejected_count: usize,
    pub metadata_survived_count: usize,
    pub metadata_true_count: usize,
    pub metadata_maybe_count: usize,
    pub metadata_rejected_by_feed_count: usize,
    pub metadata_rejected_by_flags_count: usize,
    pub metadata_rejected_by_status_count: usize,
    pub metadata_rejected_by_deleted_count: usize,
    pub metadata_rejected_by_visibility_count: usize,
    pub metadata_rejected_by_geo_count: usize,
    pub metadata_rejected_by_time_count: usize,
    pub residual_checked_count: usize,
    pub residual_passed_count: usize,
    pub residual_failed_count: usize,
    pub hypothetical_reject_feed_count: usize,
    pub hypothetical_reject_flags_count: usize,
    pub hypothetical_reject_status_count: usize,
    pub hypothetical_reject_deleted_count: usize,
    pub hypothetical_reject_visibility_count: usize,
    pub hypothetical_reject_geo_count: usize,
    pub hypothetical_reject_time_count: usize,
    pub hypothetical_reject_feed_flags_count: usize,
    pub hypothetical_reject_feed_flags_geo_count: usize,
    pub hypothetical_reject_feed_flags_geo_time_count: usize,
    pub metadata_eval_ns: u128,
    pub metadata_decode_ns: u128,
    pub metadata_supported_qual_count: usize,
    pub metadata_unsupported_qual_count: usize,
    pub metadata_all_quals_covered: bool,
    pub metadata_unavailable_param_count: usize,
    pub heap_prefilter_after_metadata_count: usize,
    pub heap_prefilter_avoided_count: usize,
    pub metadata_false_negative_count_debug: usize,
    pub block_summary_checked_count: usize,
    pub block_summary_rejected_count: usize,
    pub block_summary_maybe_count: usize,
    pub block_summary_candidates_skipped_count: usize,
    pub prefetch_next_ns: u128,
    pub index_vector_read_ns: u128,
    pub index_distance_ns: u128,
    pub index_vector_pages: usize,
    pub heap_fetch_ns: u128,
    pub heap_distance_ns: u128,
    pub prefilter_fetch_ns: u128,
    pub prefilter_filter_ns: u128,
    pub prefilter_checked_count: usize,
    pub prefilter_passed_count: usize,
    pub prefilter_window_ns: u128,
    pub prefilter_window_count: usize,
    pub prefilter_window_candidates: usize,
    pub prefilter_window_heap_blocks: usize,
    pub prefilter_window_passed: usize,
    pub vector_window_ns: u128,
    pub vector_window_count: usize,
    pub vector_window_candidates: usize,
    pub vector_window_pages: usize,
    pub vector_window_unique_pages: usize,
}

impl SearchInstrumentation {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(SearchInstrumentationInner {
                started_at: Instant::now(),
                candidate_count: Cell::new(0),
                candidate_count_before_block_prune: Cell::new(0),
                candidate_count_after_block_prune: Cell::new(0),
                emitted_count: Cell::new(0),
                build_ns: Cell::new(0),
                rerank_next_ns: Cell::new(0),
                metadata_checked_count: Cell::new(0),
                metadata_rejected_count: Cell::new(0),
                metadata_survived_count: Cell::new(0),
                metadata_true_count: Cell::new(0),
                metadata_maybe_count: Cell::new(0),
                metadata_rejected_by_feed_count: Cell::new(0),
                metadata_rejected_by_flags_count: Cell::new(0),
                metadata_rejected_by_status_count: Cell::new(0),
                metadata_rejected_by_deleted_count: Cell::new(0),
                metadata_rejected_by_visibility_count: Cell::new(0),
                metadata_rejected_by_geo_count: Cell::new(0),
                metadata_rejected_by_time_count: Cell::new(0),
                residual_checked_count: Cell::new(0),
                residual_passed_count: Cell::new(0),
                residual_failed_count: Cell::new(0),
                hypothetical_reject_feed_count: Cell::new(0),
                hypothetical_reject_flags_count: Cell::new(0),
                hypothetical_reject_status_count: Cell::new(0),
                hypothetical_reject_deleted_count: Cell::new(0),
                hypothetical_reject_visibility_count: Cell::new(0),
                hypothetical_reject_geo_count: Cell::new(0),
                hypothetical_reject_time_count: Cell::new(0),
                hypothetical_reject_feed_flags_count: Cell::new(0),
                hypothetical_reject_feed_flags_geo_count: Cell::new(0),
                hypothetical_reject_feed_flags_geo_time_count: Cell::new(0),
                metadata_eval_ns: Cell::new(0),
                metadata_decode_ns: Cell::new(0),
                metadata_supported_qual_count: Cell::new(0),
                metadata_unsupported_qual_count: Cell::new(0),
                metadata_all_quals_covered: Cell::new(false),
                metadata_unavailable_param_count: Cell::new(0),
                heap_prefilter_after_metadata_count: Cell::new(0),
                heap_prefilter_avoided_count: Cell::new(0),
                metadata_false_negative_count_debug: Cell::new(0),
                block_summary_checked_count: Cell::new(0),
                block_summary_rejected_count: Cell::new(0),
                block_summary_maybe_count: Cell::new(0),
                block_summary_candidates_skipped_count: Cell::new(0),
                rerank: vchordrq::RerankInstrumentation::new(),
                logged: Cell::new(false),
            }),
        }
    }

    pub fn rerank_instrumentation(&self) -> vchordrq::RerankInstrumentation {
        self.inner.rerank.clone()
    }

    pub fn set_candidate_count(&self, count: usize) {
        self.inner.candidate_count.set(count);
    }

    pub fn set_default_search_stats(&self, stats: vchordrq::DefaultSearchStats) {
        self.inner.candidate_count.set(stats.candidate_count);
        self.inner
            .candidate_count_before_block_prune
            .set(stats.candidate_count_before_block_prune);
        self.inner
            .candidate_count_after_block_prune
            .set(stats.candidate_count_after_block_prune);
        self.inner
            .block_summary_checked_count
            .set(self.inner.block_summary_checked_count.get() + stats.block_summary_checked_count);
        self.inner.block_summary_rejected_count.set(
            self.inner.block_summary_rejected_count.get() + stats.block_summary_rejected_count,
        );
        self.inner
            .block_summary_maybe_count
            .set(self.inner.block_summary_maybe_count.get() + stats.block_summary_maybe_count);
        self.inner.block_summary_candidates_skipped_count.set(
            self.inner.block_summary_candidates_skipped_count.get()
                + stats.block_summary_candidates_skipped_count,
        );
    }

    pub fn add_build_time(&self, duration: Duration) {
        self.inner
            .build_ns
            .set(self.inner.build_ns.get() + duration.as_nanos());
    }

    pub fn add_rerank_next_time(&self, duration: Duration) {
        self.inner
            .rerank_next_ns
            .set(self.inner.rerank_next_ns.get() + duration.as_nanos());
    }

    pub fn set_metadata_qual_stats(
        &self,
        supported: usize,
        unsupported: usize,
        all_covered: bool,
        unavailable_params: usize,
    ) {
        self.inner.metadata_supported_qual_count.set(supported);
        self.inner.metadata_unsupported_qual_count.set(unsupported);
        self.inner.metadata_all_quals_covered.set(all_covered);
        self.inner
            .metadata_unavailable_param_count
            .set(unavailable_params);
    }

    pub fn increment_metadata_checked(&self) {
        self.inner
            .metadata_checked_count
            .set(self.inner.metadata_checked_count.get() + 1);
    }

    pub fn increment_metadata_rejected(&self) {
        self.inner
            .metadata_rejected_count
            .set(self.inner.metadata_rejected_count.get() + 1);
    }

    pub fn increment_metadata_rejected_by(&self, kind: MetadataColumnKind) {
        let counter = match kind {
            MetadataColumnKind::Feed => &self.inner.metadata_rejected_by_feed_count,
            MetadataColumnKind::Flags => &self.inner.metadata_rejected_by_flags_count,
            MetadataColumnKind::Status => &self.inner.metadata_rejected_by_status_count,
            MetadataColumnKind::Deleted => &self.inner.metadata_rejected_by_deleted_count,
            MetadataColumnKind::Visibility => &self.inner.metadata_rejected_by_visibility_count,
            MetadataColumnKind::Geo => &self.inner.metadata_rejected_by_geo_count,
            MetadataColumnKind::Time => &self.inner.metadata_rejected_by_time_count,
            MetadataColumnKind::Other => return,
        };
        counter.set(counter.get() + 1);
    }

    pub fn increment_metadata_survived(&self) {
        self.inner
            .metadata_survived_count
            .set(self.inner.metadata_survived_count.get() + 1);
    }

    pub fn increment_metadata_true(&self) {
        self.inner
            .metadata_true_count
            .set(self.inner.metadata_true_count.get() + 1);
    }

    pub fn increment_metadata_maybe(&self) {
        self.inner
            .metadata_maybe_count
            .set(self.inner.metadata_maybe_count.get() + 1);
    }

    pub fn add_metadata_eval_time(&self, duration: Duration) {
        self.inner
            .metadata_eval_ns
            .set(self.inner.metadata_eval_ns.get() + duration.as_nanos());
    }

    pub fn increment_heap_prefilter_after_metadata(&self) {
        self.inner
            .heap_prefilter_after_metadata_count
            .set(self.inner.heap_prefilter_after_metadata_count.get() + 1);
    }

    pub fn increment_heap_prefilter_avoided(&self) {
        self.inner
            .heap_prefilter_avoided_count
            .set(self.inner.heap_prefilter_avoided_count.get() + 1);
    }

    pub fn increment_metadata_false_negative_debug(&self) {
        self.inner
            .metadata_false_negative_count_debug
            .set(self.inner.metadata_false_negative_count_debug.get() + 1);
    }

    pub fn add_block_summary_checked(&self) {
        self.inner
            .block_summary_checked_count
            .set(self.inner.block_summary_checked_count.get() + 1);
    }

    pub fn add_block_summary_rejected(&self, skipped: usize) {
        self.inner
            .block_summary_rejected_count
            .set(self.inner.block_summary_rejected_count.get() + 1);
        self.inner
            .block_summary_candidates_skipped_count
            .set(self.inner.block_summary_candidates_skipped_count.get() + skipped);
    }

    pub fn add_block_summary_maybe(&self) {
        self.inner
            .block_summary_maybe_count
            .set(self.inner.block_summary_maybe_count.get() + 1);
    }

    pub fn record_residual_result(&self, passed: bool) {
        self.inner
            .residual_checked_count
            .set(self.inner.residual_checked_count.get() + 1);
        if passed {
            self.inner
                .residual_passed_count
                .set(self.inner.residual_passed_count.get() + 1);
        } else {
            self.inner
                .residual_failed_count
                .set(self.inner.residual_failed_count.get() + 1);
        }
    }

    pub fn increment_hypothetical_reject_by(&self, kind: MetadataColumnKind) {
        let counter = match kind {
            MetadataColumnKind::Feed => &self.inner.hypothetical_reject_feed_count,
            MetadataColumnKind::Flags => &self.inner.hypothetical_reject_flags_count,
            MetadataColumnKind::Status => &self.inner.hypothetical_reject_status_count,
            MetadataColumnKind::Deleted => &self.inner.hypothetical_reject_deleted_count,
            MetadataColumnKind::Visibility => &self.inner.hypothetical_reject_visibility_count,
            MetadataColumnKind::Geo => &self.inner.hypothetical_reject_geo_count,
            MetadataColumnKind::Time => &self.inner.hypothetical_reject_time_count,
            MetadataColumnKind::Other => return,
        };
        counter.set(counter.get() + 1);
    }

    pub fn increment_hypothetical_feed_flags(&self) {
        self.inner
            .hypothetical_reject_feed_flags_count
            .set(self.inner.hypothetical_reject_feed_flags_count.get() + 1);
    }

    pub fn increment_hypothetical_feed_flags_geo(&self) {
        self.inner
            .hypothetical_reject_feed_flags_geo_count
            .set(self.inner.hypothetical_reject_feed_flags_geo_count.get() + 1);
    }

    pub fn increment_hypothetical_feed_flags_geo_time(&self) {
        self.inner
            .hypothetical_reject_feed_flags_geo_time_count
            .set(
                self.inner
                    .hypothetical_reject_feed_flags_geo_time_count
                    .get()
                    + 1,
            );
    }

    pub fn add_prefilter_fetch_time(&self, duration: Duration) {
        self.inner.rerank.add_prefilter_fetch_time(duration);
    }

    pub fn add_prefilter_filter_time(&self, duration: Duration) {
        self.inner.rerank.add_prefilter_filter_time(duration);
    }

    pub fn increment_prefilter_checked(&self) {
        self.inner.rerank.increment_prefilter_checked();
    }

    pub fn increment_prefilter_passed(&self) {
        self.inner.rerank.increment_prefilter_passed();
    }

    pub fn add_prefilter_window(
        &self,
        duration: Duration,
        candidates: usize,
        heap_blocks: usize,
        passed: usize,
    ) {
        self.inner
            .rerank
            .add_prefilter_window(duration, candidates, heap_blocks, passed);
    }

    pub fn increment_emitted(&self) {
        self.inner
            .emitted_count
            .set(self.inner.emitted_count.get() + 1);
    }

    pub fn mark_logged(&self) -> bool {
        !self.inner.logged.replace(true)
    }

    pub fn snapshot(&self) -> SearchInstrumentationSnapshot {
        let rerank = self.inner.rerank.snapshot();
        SearchInstrumentationSnapshot {
            candidate_count: self.inner.candidate_count.get(),
            candidate_count_before_block_prune: self.inner.candidate_count_before_block_prune.get(),
            candidate_count_after_block_prune: self.inner.candidate_count_after_block_prune.get(),
            scored_count: rerank.scored_count,
            emitted_count: self.inner.emitted_count.get(),
            build_ns: self.inner.build_ns.get(),
            rerank_next_ns: self.inner.rerank_next_ns.get(),
            total_ns: self.inner.started_at.elapsed().as_nanos(),
            metadata_checked_count: self.inner.metadata_checked_count.get(),
            metadata_rejected_count: self.inner.metadata_rejected_count.get(),
            metadata_survived_count: self.inner.metadata_survived_count.get(),
            metadata_true_count: self.inner.metadata_true_count.get(),
            metadata_maybe_count: self.inner.metadata_maybe_count.get(),
            metadata_rejected_by_feed_count: self.inner.metadata_rejected_by_feed_count.get(),
            metadata_rejected_by_flags_count: self.inner.metadata_rejected_by_flags_count.get(),
            metadata_rejected_by_status_count: self.inner.metadata_rejected_by_status_count.get(),
            metadata_rejected_by_deleted_count: self.inner.metadata_rejected_by_deleted_count.get(),
            metadata_rejected_by_visibility_count: self
                .inner
                .metadata_rejected_by_visibility_count
                .get(),
            metadata_rejected_by_geo_count: self.inner.metadata_rejected_by_geo_count.get(),
            metadata_rejected_by_time_count: self.inner.metadata_rejected_by_time_count.get(),
            residual_checked_count: self.inner.residual_checked_count.get(),
            residual_passed_count: self.inner.residual_passed_count.get(),
            residual_failed_count: self.inner.residual_failed_count.get(),
            hypothetical_reject_feed_count: self.inner.hypothetical_reject_feed_count.get(),
            hypothetical_reject_flags_count: self.inner.hypothetical_reject_flags_count.get(),
            hypothetical_reject_status_count: self.inner.hypothetical_reject_status_count.get(),
            hypothetical_reject_deleted_count: self.inner.hypothetical_reject_deleted_count.get(),
            hypothetical_reject_visibility_count: self
                .inner
                .hypothetical_reject_visibility_count
                .get(),
            hypothetical_reject_geo_count: self.inner.hypothetical_reject_geo_count.get(),
            hypothetical_reject_time_count: self.inner.hypothetical_reject_time_count.get(),
            hypothetical_reject_feed_flags_count: self
                .inner
                .hypothetical_reject_feed_flags_count
                .get(),
            hypothetical_reject_feed_flags_geo_count: self
                .inner
                .hypothetical_reject_feed_flags_geo_count
                .get(),
            hypothetical_reject_feed_flags_geo_time_count: self
                .inner
                .hypothetical_reject_feed_flags_geo_time_count
                .get(),
            metadata_eval_ns: self.inner.metadata_eval_ns.get(),
            metadata_decode_ns: self.inner.metadata_decode_ns.get(),
            metadata_supported_qual_count: self.inner.metadata_supported_qual_count.get(),
            metadata_unsupported_qual_count: self.inner.metadata_unsupported_qual_count.get(),
            metadata_all_quals_covered: self.inner.metadata_all_quals_covered.get(),
            metadata_unavailable_param_count: self.inner.metadata_unavailable_param_count.get(),
            heap_prefilter_after_metadata_count: self
                .inner
                .heap_prefilter_after_metadata_count
                .get(),
            heap_prefilter_avoided_count: self.inner.heap_prefilter_avoided_count.get(),
            metadata_false_negative_count_debug: self
                .inner
                .metadata_false_negative_count_debug
                .get(),
            block_summary_checked_count: self.inner.block_summary_checked_count.get(),
            block_summary_rejected_count: self.inner.block_summary_rejected_count.get(),
            block_summary_maybe_count: self.inner.block_summary_maybe_count.get(),
            block_summary_candidates_skipped_count: self
                .inner
                .block_summary_candidates_skipped_count
                .get(),
            prefetch_next_ns: rerank.prefetch_next_ns,
            index_vector_read_ns: rerank.index_vector_read_ns,
            index_distance_ns: rerank.index_distance_ns,
            index_vector_pages: rerank.index_vector_pages,
            heap_fetch_ns: rerank.heap_fetch_ns,
            heap_distance_ns: rerank.heap_distance_ns,
            prefilter_fetch_ns: rerank.prefilter_fetch_ns,
            prefilter_filter_ns: rerank.prefilter_filter_ns,
            prefilter_checked_count: rerank.prefilter_checked_count,
            prefilter_passed_count: rerank.prefilter_passed_count,
            prefilter_window_ns: rerank.prefilter_window_ns,
            prefilter_window_count: rerank.prefilter_window_count,
            prefilter_window_candidates: rerank.prefilter_window_candidates,
            prefilter_window_heap_blocks: rerank.prefilter_window_heap_blocks,
            prefilter_window_passed: rerank.prefilter_window_passed,
            vector_window_ns: rerank.vector_window_ns,
            vector_window_count: rerank.vector_window_count,
            vector_window_candidates: rerank.vector_window_candidates,
            vector_window_pages: rerank.vector_window_pages,
            vector_window_unique_pages: rerank.vector_window_unique_pages,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetadataPrefilterOptions {
    pub mode: MetadataPrefilterMode,
    pub schema_cols: usize,
    pub predicates: Vec<MetadataPredicate>,
    pub hypothetical_predicates: Vec<MetadataPredicate>,
    pub all_quals_covered: bool,
    pub block_prune: bool,
    pub debug: bool,
}

impl MetadataPrefilterOptions {
    pub fn can_skip_heap_prefilter(&self) -> bool {
        self.mode == MetadataPrefilterMode::CoveredSkipHeap
            && self.all_quals_covered
            && !self.predicates.is_empty()
            && self
                .predicates
                .iter()
                .all(MetadataPredicate::is_exact_for_heap_skip)
    }
}

#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub epsilon: f32,
    pub probes: Vec<u32>,
    pub max_scan_tuples: Option<u32>,
    pub maxsim_refine: u32,
    pub maxsim_threshold: u32,
    pub io_search: Io,
    pub io_rerank: Io,
    pub prefilter: bool,
    pub read_stream_batch: bool,
    pub prefilter_window: usize,
    pub vector_read_window: usize,
    pub metadata_prefilter: MetadataPrefilterOptions,
    pub instrumentation: Option<SearchInstrumentation>,
}
