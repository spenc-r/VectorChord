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
// Copyright (c) 2025-2026 TensorChord Inc.

use crate::closure_lifetime_binder::id_4;
use crate::operator::*;
use crate::tuples::{MetaTuple, WithReader};
use crate::{CandidateMetadata, RerankMethod, vectors};
use always_equal::AlwaysEqual;
use distance::Distance;
use index::fetch::{BorrowedIter, Fetch};
use index::packed::PackedRefMut;
use index::prefetcher::Prefetcher;
use index::relation::{Page, RelationRead};
use index_accessor::{Accessor2, DefaultWithDimension, LTryAccess};
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::num::NonZero;
use std::rc::Rc;
use std::time::Duration;
use vector::{VectorBorrowed, VectorOwned};

type Result = (Reverse<Distance>, AlwaysEqual<NonZero<u64>>);

pub trait RerankCandidate<'b> {
    fn payload_head(&mut self) -> (NonZero<u64>, u16);
}

impl<'b> RerankCandidate<'b> for (NonZero<u64>, u16, BorrowedIter<'b>) {
    fn payload_head(&mut self) -> (NonZero<u64>, u16) {
        (self.0, self.1)
    }
}

impl<'b> RerankCandidate<'b> for (NonZero<u64>, u16, CandidateMetadata, BorrowedIter<'b>) {
    fn payload_head(&mut self) -> (NonZero<u64>, u16) {
        (self.0, self.1)
    }
}

pub fn how(index: &impl RelationRead) -> RerankMethod {
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let rerank_in_heap = meta_tuple.rerank_in_heap();
    if rerank_in_heap {
        RerankMethod::Heap
    } else {
        RerankMethod::Index
    }
}

#[derive(Debug, Clone)]
pub struct RerankInstrumentation {
    inner: Rc<RerankInstrumentationInner>,
}

#[derive(Debug)]
struct RerankInstrumentationInner {
    scored_count: Cell<usize>,
    prefetch_next_ns: Cell<u128>,
    index_vector_read_ns: Cell<u128>,
    index_distance_ns: Cell<u128>,
    index_vector_pages: Cell<usize>,
    heap_fetch_ns: Cell<u128>,
    heap_distance_ns: Cell<u128>,
    prefilter_fetch_ns: Cell<u128>,
    prefilter_filter_ns: Cell<u128>,
    prefilter_checked_count: Cell<usize>,
    prefilter_passed_count: Cell<usize>,
    prefilter_window_ns: Cell<u128>,
    prefilter_window_count: Cell<usize>,
    prefilter_window_candidates: Cell<usize>,
    prefilter_window_heap_blocks: Cell<usize>,
    prefilter_window_passed: Cell<usize>,
    vector_window_ns: Cell<u128>,
    vector_window_count: Cell<usize>,
    vector_window_candidates: Cell<usize>,
    vector_window_pages: Cell<usize>,
    vector_window_unique_pages: Cell<usize>,
}

#[derive(Debug, Clone, Copy)]
pub struct RerankInstrumentationSnapshot {
    pub scored_count: usize,
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

impl RerankInstrumentation {
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RerankInstrumentationInner {
                scored_count: Cell::new(0),
                prefetch_next_ns: Cell::new(0),
                index_vector_read_ns: Cell::new(0),
                index_distance_ns: Cell::new(0),
                index_vector_pages: Cell::new(0),
                heap_fetch_ns: Cell::new(0),
                heap_distance_ns: Cell::new(0),
                prefilter_fetch_ns: Cell::new(0),
                prefilter_filter_ns: Cell::new(0),
                prefilter_checked_count: Cell::new(0),
                prefilter_passed_count: Cell::new(0),
                prefilter_window_ns: Cell::new(0),
                prefilter_window_count: Cell::new(0),
                prefilter_window_candidates: Cell::new(0),
                prefilter_window_heap_blocks: Cell::new(0),
                prefilter_window_passed: Cell::new(0),
                vector_window_ns: Cell::new(0),
                vector_window_count: Cell::new(0),
                vector_window_candidates: Cell::new(0),
                vector_window_pages: Cell::new(0),
                vector_window_unique_pages: Cell::new(0),
            }),
        }
    }

    pub fn increment_scored(&self) {
        self.inner
            .scored_count
            .set(self.inner.scored_count.get() + 1);
    }

    pub fn add_prefetch_next_time(&self, duration: Duration) {
        self.inner
            .prefetch_next_ns
            .set(self.inner.prefetch_next_ns.get() + duration.as_nanos());
    }

    pub fn add_index_vector_read_time(&self, duration: Duration) {
        self.inner
            .index_vector_read_ns
            .set(self.inner.index_vector_read_ns.get() + duration.as_nanos());
    }

    pub fn add_index_distance_time(&self, duration: Duration) {
        self.inner
            .index_distance_ns
            .set(self.inner.index_distance_ns.get() + duration.as_nanos());
    }

    pub fn increment_index_vector_pages(&self) {
        self.inner
            .index_vector_pages
            .set(self.inner.index_vector_pages.get() + 1);
    }

    pub fn add_heap_fetch_time(&self, duration: Duration) {
        self.inner
            .heap_fetch_ns
            .set(self.inner.heap_fetch_ns.get() + duration.as_nanos());
    }

    pub fn add_heap_distance_time(&self, duration: Duration) {
        self.inner
            .heap_distance_ns
            .set(self.inner.heap_distance_ns.get() + duration.as_nanos());
    }

    pub fn add_prefilter_fetch_time(&self, duration: Duration) {
        self.inner
            .prefilter_fetch_ns
            .set(self.inner.prefilter_fetch_ns.get() + duration.as_nanos());
    }

    pub fn add_prefilter_filter_time(&self, duration: Duration) {
        self.inner
            .prefilter_filter_ns
            .set(self.inner.prefilter_filter_ns.get() + duration.as_nanos());
    }

    pub fn increment_prefilter_checked(&self) {
        self.inner
            .prefilter_checked_count
            .set(self.inner.prefilter_checked_count.get() + 1);
    }

    pub fn increment_prefilter_passed(&self) {
        self.inner
            .prefilter_passed_count
            .set(self.inner.prefilter_passed_count.get() + 1);
    }

    pub fn add_prefilter_window(
        &self,
        duration: Duration,
        candidates: usize,
        heap_blocks: usize,
        passed: usize,
    ) {
        self.inner
            .prefilter_window_ns
            .set(self.inner.prefilter_window_ns.get() + duration.as_nanos());
        self.inner
            .prefilter_window_count
            .set(self.inner.prefilter_window_count.get() + 1);
        self.inner
            .prefilter_window_candidates
            .set(self.inner.prefilter_window_candidates.get() + candidates);
        self.inner
            .prefilter_window_heap_blocks
            .set(self.inner.prefilter_window_heap_blocks.get() + heap_blocks);
        self.inner
            .prefilter_window_passed
            .set(self.inner.prefilter_window_passed.get() + passed);
    }

    pub fn add_vector_window(
        &self,
        duration: Duration,
        candidates: usize,
        pages: usize,
        unique_pages: usize,
    ) {
        self.inner
            .vector_window_ns
            .set(self.inner.vector_window_ns.get() + duration.as_nanos());
        self.inner
            .vector_window_count
            .set(self.inner.vector_window_count.get() + 1);
        self.inner
            .vector_window_candidates
            .set(self.inner.vector_window_candidates.get() + candidates);
        self.inner
            .vector_window_pages
            .set(self.inner.vector_window_pages.get() + pages);
        self.inner
            .vector_window_unique_pages
            .set(self.inner.vector_window_unique_pages.get() + unique_pages);
    }

    pub fn snapshot(&self) -> RerankInstrumentationSnapshot {
        RerankInstrumentationSnapshot {
            scored_count: self.inner.scored_count.get(),
            prefetch_next_ns: self.inner.prefetch_next_ns.get(),
            index_vector_read_ns: self.inner.index_vector_read_ns.get(),
            index_distance_ns: self.inner.index_distance_ns.get(),
            index_vector_pages: self.inner.index_vector_pages.get(),
            heap_fetch_ns: self.inner.heap_fetch_ns.get(),
            heap_distance_ns: self.inner.heap_distance_ns.get(),
            prefilter_fetch_ns: self.inner.prefilter_fetch_ns.get(),
            prefilter_filter_ns: self.inner.prefilter_filter_ns.get(),
            prefilter_checked_count: self.inner.prefilter_checked_count.get(),
            prefilter_passed_count: self.inner.prefilter_passed_count.get(),
            prefilter_window_ns: self.inner.prefilter_window_ns.get(),
            prefilter_window_count: self.inner.prefilter_window_count.get(),
            prefilter_window_candidates: self.inner.prefilter_window_candidates.get(),
            prefilter_window_heap_blocks: self.inner.prefilter_window_heap_blocks.get(),
            prefilter_window_passed: self.inner.prefilter_window_passed.get(),
            vector_window_ns: self.inner.vector_window_ns.get(),
            vector_window_count: self.inner.vector_window_count.get(),
            vector_window_candidates: self.inner.vector_window_candidates.get(),
            vector_window_pages: self.inner.vector_window_pages.get(),
            vector_window_unique_pages: self.inner.vector_window_unique_pages.get(),
        }
    }
}

pub struct Reranker<T, F, P, W> {
    prefetcher: P,
    cache: BinaryHeap<Result>,
    f: F,
    instrumentation: Option<RerankInstrumentation>,
    _phantom: PhantomData<fn(T, W) -> (T, W)>,
}

impl<'b, T, F, P, W> Iterator for Reranker<T, F, P, W>
where
    F: FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut,
    W::T: RerankCandidate<'b>,
    ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>): Fetch<'b>,
{
    type Item = (Distance, NonZero<u64>);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(((_, AlwaysEqual(mut w)), prefetch)) = {
            let started_at = std::time::Instant::now();
            let next = self
                .prefetcher
                .next_if(|((d, _), ..)| Some(*d) > self.cache.peek().map(|(d, ..)| *d));
            if let Some(instrumentation) = &self.instrumentation {
                instrumentation.add_prefetch_next_time(started_at.elapsed());
            }
            next
        } {
            let (payload, head) = w.get_mut().payload_head();
            if let Some(instrumentation) = &self.instrumentation {
                instrumentation.increment_scored();
            }
            if let Some(distance) = (self.f)(payload, prefetch, head) {
                self.cache.push((Reverse(distance), AlwaysEqual(payload)));
            };
        }
        let (Reverse(distance), AlwaysEqual(payload)) = self.cache.pop()?;
        Some((distance, payload))
    }
}

impl<T, F, P, W> Reranker<T, F, P, W> {
    pub fn with_optional_instrumentation(
        mut self,
        instrumentation: Option<RerankInstrumentation>,
    ) -> Self {
        self.instrumentation = instrumentation;
        self
    }

    pub fn finish(self) -> (P, impl Iterator<Item = Result>) {
        (self.prefetcher, self.cache.into_iter())
    }
}

pub fn rerank_index<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut,
>(
    vector: O::Vector,
    prefetcher: P,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W>
where
    W::T: RerankCandidate<'b>,
    ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>): Fetch<'b>,
{
    rerank_index_instrumented::<O, T, P, W>(vector, prefetcher, None)
}

pub fn rerank_index_instrumented<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut,
>(
    vector: O::Vector,
    prefetcher: P,
    instrumentation: Option<RerankInstrumentation>,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W>
where
    W::T: RerankCandidate<'b>,
    ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>): Fetch<'b>,
{
    let dim = vector.as_borrowed().dim();
    let instrumentation_for_f = instrumentation.clone();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        instrumentation,
        f: id_4::<_, P, _, _, _>(move |payload, prefetch, head| {
            vectors::read_with_instrumentation::<P::R, O, _>(
                prefetch,
                head,
                payload,
                LTryAccess::new(
                    O::Vector::unpack(vector.as_borrowed()),
                    O::DistanceAccessor::default_with_dimension(dim),
                ),
                instrumentation_for_f.as_ref(),
            )
        }),
        _phantom: PhantomData,
    }
}

pub fn rerank_heap<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut,
>(
    vector: O::Vector,
    prefetcher: P,
    fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W>
where
    W::T: RerankCandidate<'b>,
    ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>): Fetch<'b>,
{
    rerank_heap_instrumented::<O, T, P, W>(vector, prefetcher, fetch, None)
}

pub fn rerank_heap_instrumented<
    'b,
    O: Operator,
    T,
    P: Prefetcher<'b, Item = ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>)>,
    W: 'b + PackedRefMut,
>(
    vector: O::Vector,
    prefetcher: P,
    mut fetch: impl FnMut(NonZero<u64>) -> Option<O::Vector> + 'b,
    instrumentation: Option<RerankInstrumentation>,
) -> Reranker<T, impl FnMut(NonZero<u64>, P::Guards, u16) -> Option<Distance>, P, W>
where
    W::T: RerankCandidate<'b>,
    ((Reverse<Distance>, AlwaysEqual<T>), AlwaysEqual<W>): Fetch<'b>,
{
    let dim = vector.as_borrowed().dim();
    let instrumentation_for_f = instrumentation.clone();
    Reranker {
        prefetcher,
        cache: BinaryHeap::new(),
        instrumentation,
        f: id_4::<_, P, _, _, _>(move |payload, _, _| {
            let unpack = O::Vector::unpack(vector.as_borrowed());
            let started_at = std::time::Instant::now();
            let vector = fetch(payload)?;
            if let Some(instrumentation) = &instrumentation_for_f {
                instrumentation.add_heap_fetch_time(started_at.elapsed());
            }
            let vector = O::Vector::unpack(vector.as_borrowed());
            let mut accessor = O::DistanceAccessor::default_with_dimension(dim);
            let started_at = std::time::Instant::now();
            accessor.push(unpack.0, vector.0);
            let distance = accessor.finish(unpack.1, vector.1);
            if let Some(instrumentation) = &instrumentation_for_f {
                instrumentation.add_heap_distance_time(started_at.elapsed());
            }
            Some(distance)
        }),
        _phantom: PhantomData,
    }
}
