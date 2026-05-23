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

use crate::index::fetcher::*;
use crate::index::gucs::MetadataPrefilterMode;
use crate::index::opclass::Sphere;
use crate::index::scanners::{Io, SearchBuilder};
use crate::index::vchordrq::dispatch::*;
use crate::index::vchordrq::filter::{HeapBlockKey, WindowFilterStats, window_filter};
use crate::index::vchordrq::opclass::Opfamily;
use crate::index::vchordrq::scanners::{
    MetadataPrefilterOptions, SearchInstrumentation, SearchOptions,
};
use crate::recorder::{Recorder, text};
use always_equal::AlwaysEqual;
use dary_heap::QuaternaryHeap as Heap;
use distance::Distance;
use index::accessor::{Dot, L2S};
use index::bump::Bump;
use index::fetch::BorrowedIter;
use index::packed::{PackedRefMut, PackedRefMut4};
use index::prefetcher::*;
use index::relation::{Hints, Page, RelationPrefetch, RelationRead, RelationReadStream};
use simd::f16;
use std::cmp::Reverse;
use std::num::NonZero;
use std::time::Instant;
use vchordrq::types::{DistanceKind, OwnedVector, VectorKind};
use vchordrq::{
    BlockPrune, CandidateMetadata, RerankMethod,
    default_search_with_candidate_filter_and_block_prune, how, rerank_heap_instrumented,
    rerank_index_instrumented,
};
use vector::VectorOwned;
use vector::rabitq4::Rabitq4Owned;
use vector::rabitq8::Rabitq8Owned;
use vector::vect::VectOwned;

type CandidateItem<'b> = (
    (Reverse<Distance>, AlwaysEqual<()>),
    AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, CandidateMetadata, BorrowedIter<'b>)>>,
);

impl HeapBlockKey for ([u16; 3], NonZero<u64>, CandidateMetadata) {
    fn heap_block_key(self) -> [u16; 3] {
        self.0
    }
}

fn candidate_heap_key(item: &CandidateItem<'_>) -> ([u16; 3], NonZero<u64>, CandidateMetadata) {
    let packed = &item.1.0;
    let (pointer, _, metadata, _) = packed.get();
    let (key, _) = pointer_to_kv(*pointer);
    (key, *pointer, *metadata)
}

pub struct DefaultBuilder {
    opfamily: Opfamily,
    orderbys: Vec<Option<OwnedVector>>,
    spheres: Vec<Option<Sphere<OwnedVector>>>,
}

fn instrumented_default_search<'b, R, O>(
    instrumentation: Option<&SearchInstrumentation>,
    metadata_prefilter: MetadataPrefilterOptions,
    _fetcher: &mut impl Fetcher,
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    probes: Vec<u32>,
    epsilon: f32,
    bump: &'b impl Bump,
    prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
) -> Vec<(
    (Reverse<Distance>, AlwaysEqual<()>),
    AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, CandidateMetadata, BorrowedIter<'b>)>>,
)>
where
    R: RelationRead,
    R::Page: Page<Opaque = vchordrq::Opaque>,
    O: vchordrq::operator::Operator,
{
    let started_at = Instant::now();
    let block_prune_enabled = metadata_prefilter.block_prune
        && metadata_prefilter.mode != MetadataPrefilterMode::Off
        && !metadata_prefilter.predicates.is_empty();
    let (results, search_stats) = default_search_with_candidate_filter_and_block_prune::<R, O>(
        index,
        vector,
        probes,
        epsilon,
        bump,
        prefetch_h1_vectors,
        prefetch_h0_tuples,
        |_, _| true,
        |candidate_metadata, payload| {
            if !block_prune_enabled {
                return BlockPrune::Disabled;
            }
            if block_summary_rejects(candidate_metadata, payload, &metadata_prefilter) {
                BlockPrune::Skip
            } else {
                BlockPrune::Scan
            }
        },
    );
    if let Some(instrumentation) = instrumentation {
        instrumentation.add_build_time(started_at.elapsed());
        instrumentation.set_default_search_stats(search_stats);
    }
    results
}

fn block_summary_rejects(
    candidate_metadata: &[CandidateMetadata; 32],
    payload: &[Option<NonZero<u64>>; 32],
    metadata_prefilter: &MetadataPrefilterOptions,
) -> bool {
    let mut saw_candidate = false;
    for (metadata, payload) in candidate_metadata.iter().zip(payload.iter()) {
        if payload.is_none() {
            continue;
        }
        saw_candidate = true;
        if candidate_metadata_definitely_rejected(*metadata, &metadata_prefilter.predicates)
            .is_none()
        {
            return false;
        }
    }
    saw_candidate
}

fn candidate_metadata_definitely_rejected(
    candidate_metadata: CandidateMetadata,
    predicates: &[crate::index::vchordrq::am::metadata_qual::MetadataPredicate],
) -> Option<&str> {
    for predicate in predicates {
        match predicate.is_definitely_false(candidate_metadata) {
            Some(true) => return Some(&predicate.column_name),
            Some(false) => {}
            None => return None,
        }
    }
    None
}

fn metadata_candidate_allows(
    payload: NonZero<u64>,
    candidate_metadata: CandidateMetadata,
    metadata_prefilter: &MetadataPrefilterOptions,
    fetcher: &mut impl Fetcher,
    instrumentation: Option<&SearchInstrumentation>,
) -> bool {
    if metadata_prefilter.mode == MetadataPrefilterMode::Off {
        return true;
    }
    if metadata_prefilter.predicates.is_empty() {
        return true;
    };
    if let Some(instrumentation) = instrumentation {
        instrumentation.increment_metadata_checked();
    }
    let started_at = Instant::now();
    let mut maybe = false;
    let mut rejected_by = None;
    for predicate in &metadata_prefilter.predicates {
        match predicate.is_definitely_false(candidate_metadata) {
            Some(true) => {
                rejected_by = Some(predicate.column_name.as_str());
                break;
            }
            Some(false) => {}
            None => {
                maybe = true;
            }
        }
    }
    if let Some(instrumentation) = instrumentation {
        instrumentation.add_metadata_eval_time(started_at.elapsed());
    }
    if let Some(rejected_by) = rejected_by {
        if let Some(instrumentation) = instrumentation {
            instrumentation.increment_metadata_rejected();
            instrumentation.increment_metadata_rejected_by(rejected_by);
            instrumentation.increment_heap_prefilter_avoided();
        }
    } else {
        if let Some(instrumentation) = instrumentation {
            instrumentation.increment_metadata_survived();
            if maybe {
                instrumentation.increment_metadata_maybe();
            } else {
                instrumentation.increment_metadata_true();
            }
        }
        return true;
    }
    let rejected_by = rejected_by.expect("metadata reject kind must be set");
    if metadata_prefilter.debug {
        let (key, _) = pointer_to_kv(payload);
        if run_prefilter(fetcher, key, None, false) {
            if let Some(instrumentation) = instrumentation {
                instrumentation.increment_metadata_false_negative_debug();
            }
            pgrx::error!(
                "vchordrq metadata prefilter false negative for column={}",
                rejected_by
            );
        }
    }
    false
}

fn run_prefilter(
    fetcher: &mut impl Fetcher,
    key: [u16; 3],
    instrumentation: Option<&SearchInstrumentation>,
    count: bool,
) -> bool {
    if count {
        if let Some(instrumentation) = instrumentation {
            instrumentation.increment_prefilter_checked();
            instrumentation.increment_heap_prefilter_after_metadata();
        }
    }
    let started_at = Instant::now();
    let Some(mut tuple) = fetcher.fetch(key) else {
        if let Some(instrumentation) = instrumentation {
            instrumentation.add_prefilter_fetch_time(started_at.elapsed());
        }
        return false;
    };
    if let Some(instrumentation) = instrumentation {
        instrumentation.add_prefilter_fetch_time(started_at.elapsed());
    }
    let started_at = Instant::now();
    let passed = tuple.filter();
    if let Some(instrumentation) = instrumentation {
        instrumentation.add_prefilter_filter_time(started_at.elapsed());
        if passed {
            instrumentation.increment_prefilter_passed();
        }
    }
    passed
}

fn instrumented_prefilter(
    fetcher: &mut impl Fetcher,
    key: [u16; 3],
    instrumentation: Option<&SearchInstrumentation>,
) -> bool {
    run_prefilter(fetcher, key, instrumentation, true)
}

fn metadata_heap_prefilter_allows(
    fetcher: &mut impl Fetcher,
    key: [u16; 3],
    candidate_metadata: CandidateMetadata,
    instrumentation: Option<&SearchInstrumentation>,
    metadata_prefilter: &MetadataPrefilterOptions,
) -> bool {
    if metadata_prefilter.can_skip_heap_prefilter(candidate_metadata) {
        if let Some(instrumentation) = instrumentation {
            instrumentation.increment_heap_prefilter_after_metadata();
            instrumentation.record_residual_result(true);
        }
        true
    } else {
        let passed = instrumented_prefilter(fetcher, key, instrumentation);
        if let Some(instrumentation) = instrumentation {
            instrumentation.record_residual_result(passed);
            if !passed {
                record_hypothetical_rejects(
                    candidate_metadata,
                    &metadata_prefilter.hypothetical_predicates,
                    instrumentation,
                );
            }
        }
        passed
    }
}

fn record_hypothetical_rejects(
    candidate_metadata: CandidateMetadata,
    predicates: &[crate::index::vchordrq::am::metadata_qual::MetadataPredicate],
    instrumentation: &SearchInstrumentation,
) {
    for predicate in predicates {
        if predicate
            .is_definitely_false(candidate_metadata)
            .unwrap_or(false)
        {
            instrumentation.increment_hypothetical_reject_by(&predicate.column_name);
        }
    }
}

fn boxed_rerank_index<'b, R, O, S, M>(
    io: Io,
    _vector_read_window: usize,
    index: &'b R,
    sequence: S,
    vector: O::Vector,
    rerank_hints: Hints,
    instrumentation: Option<vchordrq::RerankInstrumentation>,
    map: M,
) -> Box<dyn Iterator<Item = (f32, NonZero<u64>)> + 'b>
where
    R: RelationRead + RelationPrefetch + RelationReadStream + 'b,
    R::Page: Page<Opaque = vchordrq::Opaque>,
    O: vchordrq::operator::Operator + 'b,
    O::Vector: 'b,
    S: Sequence<Item = CandidateItem<'b>> + 'b,
    M: FnMut((Distance, NonZero<u64>)) -> (f32, NonZero<u64>) + 'b,
{
    match io {
        Io::Plain => {
            let prefetcher = PlainPrefetcher::new(index, sequence);
            Box::new(
                rerank_index_instrumented::<O, _, _, _>(vector, prefetcher, instrumentation)
                    .map(map),
            )
        }
        Io::Simple => {
            let prefetcher = SimplePrefetcher::new(index, sequence);
            Box::new(
                rerank_index_instrumented::<O, _, _, _>(vector, prefetcher, instrumentation)
                    .map(map),
            )
        }
        Io::Stream => {
            let prefetcher = StreamPrefetcher::new(index, sequence, rerank_hints);
            Box::new(
                rerank_index_instrumented::<O, _, _, _>(vector, prefetcher, instrumentation)
                    .map(map),
            )
        }
    }
}

impl SearchBuilder for DefaultBuilder {
    type Options = SearchOptions;

    type Opfamily = Opfamily;

    type Opaque = vchordrq::Opaque;

    fn new(opfamily: Opfamily) -> Self {
        assert!(matches!(
            opfamily,
            Opfamily::HalfvecCosine
                | Opfamily::HalfvecIp
                | Opfamily::HalfvecL2
                | Opfamily::VectorCosine
                | Opfamily::VectorIp
                | Opfamily::VectorL2
                | Opfamily::Rabitq8Cosine
                | Opfamily::Rabitq8Ip
                | Opfamily::Rabitq8L2
                | Opfamily::Rabitq4Cosine
                | Opfamily::Rabitq4Ip
                | Opfamily::Rabitq4L2
        ));
        Self {
            opfamily,
            orderbys: Vec::new(),
            spheres: Vec::new(),
        }
    }

    unsafe fn add(&mut self, strategy: u16, datum: Option<pgrx::pg_sys::Datum>) {
        match strategy {
            1 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_vector(x)) };
                self.orderbys.push(x);
            }
            2 => {
                let x = unsafe { datum.and_then(|x| self.opfamily.input_sphere(x)) };
                self.spheres.push(x);
            }
            _ => unreachable!(),
        }
    }

    fn build<'b, R>(
        self,
        index: &'b R,
        options: SearchOptions,
        mut fetcher: impl Fetcher + 'b,
        bump: &'b impl Bump,
        recorder: impl Recorder,
    ) -> Box<dyn Iterator<Item = (f32, [u16; 3], bool)> + 'b>
    where
        R: RelationRead + RelationPrefetch + RelationReadStream,
        R::Page: Page<Opaque = vchordrq::Opaque>,
    {
        let mut vector = None;
        let mut threshold = None;
        let mut recheck = false;
        for orderby_vector in self.orderbys.into_iter().flatten() {
            if vector.is_none() {
                vector = Some(orderby_vector);
            } else {
                pgrx::error!("vector search with multiple vectors is not supported");
            }
        }
        for Sphere { center, radius } in self.spheres.into_iter().flatten() {
            if vector.is_none() {
                (vector, threshold) = (Some(center), Some(radius));
            } else {
                recheck = true;
            }
        }
        let opfamily = self.opfamily;
        let Some(vector) = vector else {
            return Box::new(std::iter::empty()) as Box<dyn Iterator<Item = (f32, [u16; 3], bool)>>;
        };
        let instrumentation = options.instrumentation.clone();
        let rerank_instrumentation = instrumentation
            .as_ref()
            .map(SearchInstrumentation::rerank_instrumentation);
        let prefilter_window = options.prefilter_window;
        let vector_read_window = options.vector_read_window;
        let search_hints = Hints::default().full(true).batch(options.read_stream_batch);
        let rerank_hints = Hints::default()
            .full(false)
            .batch(options.read_stream_batch);
        let make_h1_plain_prefetcher = MakeH1PlainPrefetcher { index };
        let make_h0_plain_prefetcher = MakeH0PlainPrefetcher { index };
        let make_h0_simple_prefetcher = MakeH0SimplePrefetcher { index };
        let make_h0_stream_prefetcher = MakeH0StreamPrefetcher {
            index,
            hints: search_hints,
        };
        let f = move |(distance, payload)| (opfamily.output(distance), payload);
        let iter: Box<dyn Iterator<Item = (f32, NonZero<u64>)>> =
            match (opfamily.vector_kind(), opfamily.distance_kind()) {
                (VectorKind::Vecf32, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<VectOwned<f32>, L2S>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
                (VectorKind::Vecf32, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<VectOwned<f32>, Dot>;
                    let unprojected = if let OwnedVector::Vecf32(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf32(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
                (VectorKind::Vecf16, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<VectOwned<f16>, L2S>;
                    let unprojected = if let OwnedVector::Vecf16(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
                (VectorKind::Vecf16, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<VectOwned<f16>, Dot>;
                    let unprojected = if let OwnedVector::Vecf16(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let projected = RandomProject::project(unprojected.as_borrowed());
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            projected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw = if let OwnedVector::Vecf16(vector) = maybe_vector.unwrap()
                                {
                                    vector
                                } else {
                                    unreachable!()
                                };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
                (VectorKind::Rabitq8, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<Rabitq8Owned, L2S>;
                    let unprojected = if let OwnedVector::Rabitq8(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq8(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq8(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
                (VectorKind::Rabitq8, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<Rabitq8Owned, Dot>;
                    let unprojected = if let OwnedVector::Rabitq8(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq8(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq8(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
                (VectorKind::Rabitq4, DistanceKind::L2S) => {
                    type Op = vchordrq::operator::Op<Rabitq4Owned, L2S>;
                    let unprojected = if let OwnedVector::Rabitq4(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq4(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq4(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
                (VectorKind::Rabitq4, DistanceKind::Dot) => {
                    type Op = vchordrq::operator::Op<Rabitq4Owned, Dot>;
                    let unprojected = if let OwnedVector::Rabitq4(vector) = vector.clone() {
                        vector
                    } else {
                        unreachable!()
                    };
                    let results = match options.io_search {
                        Io::Plain => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_plain_prefetcher,
                        ),
                        Io::Simple => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_simple_prefetcher,
                        ),
                        Io::Stream => instrumented_default_search::<_, Op>(
                            instrumentation.as_ref(),
                            options.metadata_prefilter.clone(),
                            &mut fetcher,
                            index,
                            unprojected.as_borrowed(),
                            options.probes,
                            options.epsilon,
                            bump,
                            make_h1_plain_prefetcher,
                            make_h0_stream_prefetcher,
                        ),
                    };
                    let method = how(index);
                    let sequence = Heap::from(results);
                    match (method, options.io_rerank, options.prefilter) {
                        (RerankMethod::Index, Io::Plain, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Plain, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Plain,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Simple, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Simple,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, false) => {
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Index, Io::Stream, true) => {
                            let instrumentation = instrumentation.clone();
                            let window_instrumentation = instrumentation.clone();
                            let key = candidate_heap_key;
                            let metadata_prefilter = options.metadata_prefilter.clone();
                            let predicate = move |(key, payload, candidate_metadata)| {
                                metadata_candidate_allows(
                                    payload,
                                    candidate_metadata,
                                    &metadata_prefilter,
                                    &mut fetcher,
                                    instrumentation.as_ref(),
                                ) && metadata_heap_prefilter_allows(
                                    &mut fetcher,
                                    key,
                                    candidate_metadata,
                                    instrumentation.as_ref(),
                                    &metadata_prefilter,
                                )
                            };
                            let observer = move |stats: WindowFilterStats| {
                                if let Some(instrumentation) = &window_instrumentation {
                                    instrumentation.add_prefilter_window(
                                        stats.duration,
                                        stats.candidates,
                                        stats.heap_blocks,
                                        stats.passed,
                                    );
                                }
                            };
                            let sequence =
                                window_filter(sequence, prefilter_window, key, predicate, observer);
                            boxed_rerank_index::<_, Op, _, _>(
                                Io::Stream,
                                vector_read_window,
                                index,
                                sequence,
                                unprojected,
                                rerank_hints,
                                rerank_instrumentation.clone(),
                                f,
                            )
                        }
                        (RerankMethod::Heap, _, false) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq4(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                        (RerankMethod::Heap, _, true) => {
                            let fetch = move |payload| {
                                let (key, _) = pointer_to_kv(payload);
                                let mut tuple = fetcher.fetch(key)?;
                                if !tuple.filter() {
                                    return None;
                                }
                                let (datums, is_nulls) = tuple.build();
                                let datum = (!is_nulls[0]).then_some(datums[0]);
                                let maybe_vector =
                                    unsafe { datum.and_then(|x| opfamily.input_vector(x)) };
                                let raw =
                                    if let OwnedVector::Rabitq4(vector) = maybe_vector.unwrap() {
                                        vector
                                    } else {
                                        unreachable!()
                                    };
                                Some(raw)
                            };
                            let prefetcher = PlainPrefetcher::new(index, sequence);
                            Box::new(
                                rerank_heap_instrumented::<Op, _, _, _>(
                                    unprojected,
                                    prefetcher,
                                    fetch,
                                    rerank_instrumentation.clone(),
                                )
                                .map(f),
                            )
                        }
                    }
                }
            };
        let iter = if let Some(threshold) = threshold {
            Box::new(iter.take_while(move |(x, _)| *x < threshold))
        } else {
            iter
        };
        let iter = if let Some(max_scan_tuples) = options.max_scan_tuples {
            Box::new(iter.take(max_scan_tuples as _))
        } else {
            iter
        };
        if recorder.is_enabled() {
            match &vector {
                OwnedVector::Vecf32(v) => {
                    recorder.send(&text::vector_out(v.as_borrowed()));
                }
                OwnedVector::Vecf16(v) => {
                    recorder.send(&text::halfvec_out(v.as_borrowed()));
                }
                OwnedVector::Rabitq8(v) => {
                    recorder.send(&text::rabitq8_out(v.as_borrowed()));
                }
                OwnedVector::Rabitq4(v) => {
                    recorder.send(&text::rabitq4_out(v.as_borrowed()));
                }
            }
        }
        Box::new(iter.map(move |(distance, pointer)| {
            let (key, _) = pointer_to_kv(pointer);
            (distance, key, recheck)
        }))
    }
}

#[inline(always)]
pub fn id_0<F, A: ?Sized, B: ?Sized, C: ?Sized, D: ?Sized, R: ?Sized>(f: F) -> F
where
    F: for<'a> FnMut(&(A, AlwaysEqual<PackedRefMut4<'a, (B, C, D)>>)) -> R,
{
    f
}
