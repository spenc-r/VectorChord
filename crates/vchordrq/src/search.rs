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

use crate::closure_lifetime_binder::{id_0, id_1, id_5};
use crate::linked_vec::LinkedVec;
use crate::operator::*;
use crate::tape::{by_directory, by_next};
use crate::tuples::*;
use crate::{CandidateMetadata, Opaque, centroids, tape};
use always_equal::AlwaysEqual;
use distance::Distance;
use index::bump::Bump;
use index::fetch::BorrowedIter;
use index::packed::{PackedRefMut4, PackedRefMut8};
use index::prefetcher::{Prefetcher, PrefetcherHeapFamily, PrefetcherSequenceFamily};
use index::relation::{Page, RelationRead};
use index_accessor::{DefaultWithDimension, FunctionalAccessor, LAccess};
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZero;
use vector::{VectorBorrowed, VectorOwned};

type Extra1<'b> = &'b mut (u32, f32, u16, BorrowedIter<'b>);

#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultSearchStats {
    pub candidate_count: usize,
    pub candidate_count_before_block_prune: usize,
    pub candidate_count_after_block_prune: usize,
    pub block_summary_checked_count: usize,
    pub block_summary_rejected_count: usize,
    pub block_summary_maybe_count: usize,
    pub block_summary_candidates_skipped_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockPrune {
    Disabled,
    Scan,
    Skip,
}

pub fn default_search<'b, R: RelationRead, O: Operator>(
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
    R::Page: Page<Opaque = Opaque>,
{
    let (results, _) = default_search_with_candidate_filter::<R, O>(
        index,
        vector,
        probes,
        epsilon,
        bump,
        prefetch_h1_vectors,
        prefetch_h0_tuples,
        |_, _| true,
    );
    results
}

pub fn default_search_with_candidate_filter<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    probes: Vec<u32>,
    epsilon: f32,
    bump: &'b impl Bump,
    mut prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
    mut candidate_filter: impl FnMut(NonZero<u64>, CandidateMetadata) -> bool,
) -> (
    Vec<(
        (Reverse<Distance>, AlwaysEqual<()>),
        AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, CandidateMetadata, BorrowedIter<'b>)>>,
    )>,
    DefaultSearchStats,
)
where
    R::Page: Page<Opaque = Opaque>,
{
    default_search_with_candidate_filter_and_block_prune::<R, O>(
        index,
        vector,
        probes,
        epsilon,
        bump,
        prefetch_h1_vectors,
        prefetch_h0_tuples,
        |payload, metadata| candidate_filter(payload, metadata),
        |_, _| BlockPrune::Disabled,
    )
}

pub fn default_search_with_candidate_filter_and_block_prune<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    probes: Vec<u32>,
    epsilon: f32,
    bump: &'b impl Bump,
    mut prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
    mut candidate_filter: impl FnMut(NonZero<u64>, CandidateMetadata) -> bool,
    mut block_prune: impl FnMut(&[CandidateMetadata; 32], &[Option<NonZero<u64>>; 32]) -> BlockPrune,
) -> (
    Vec<(
        (Reverse<Distance>, AlwaysEqual<()>),
        AlwaysEqual<PackedRefMut4<'b, (NonZero<u64>, u16, CandidateMetadata, BorrowedIter<'b>)>>,
    )>,
    DefaultSearchStats,
)
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    let cells = meta_tuple.cells().to_vec();
    assert_eq!(dim, vector.dim(), "unmatched dimensions");
    if height_of_root as usize != 1 + probes.len() {
        panic!(
            "usage: need {} probes, but {} probes provided",
            height_of_root - 1,
            probes.len()
        );
    }
    debug_assert_eq!(cells[(height_of_root - 1) as usize], 1);

    type State = Vec<(Reverse<Distance>, AlwaysEqual<f32>, AlwaysEqual<u32>)>;
    let mut state: State = if is_residual {
        let prefetch =
            BorrowedIter::from_slice(meta_tuple.centroid_prefetch(), |x| bump.alloc_slice(x));
        let head = meta_tuple.centroid_head();
        let distance = centroids::read::<R, O, _>(
            prefetch.map(|id| index.read(id)),
            head,
            LAccess::new(
                O::Vector::unpack(vector),
                O::DistanceAccessor::default_with_dimension(dim),
            ),
        );
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        vec![(Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))]
    } else {
        // fast path
        let distance = Distance::ZERO;
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        vec![(Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))]
    };

    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);

    let mut step = |state: State| {
        let mut results = LinkedVec::<(_, AlwaysEqual<Extra1<'b>>)>::new();
        for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                |(rough, err), head, norm, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(bump.alloc((
                            first,
                            norm,
                            head,
                            BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                        ))),
                    ));
                },
            );
        }
        let mut heap = prefetch_h1_vectors.prefetch(results.into_vec());
        let mut cache = BinaryHeap::<(_, _, _)>::new();
        std::iter::from_fn(move || {
            while let Some(((Reverse(_), AlwaysEqual(&mut (first, norm, head, ..))), prefetch)) =
                heap.next_if(|(d, _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
            {
                let distance = centroids::read::<R, O, _>(
                    prefetch,
                    head,
                    LAccess::new(
                        O::Vector::unpack(vector),
                        O::DistanceAccessor::default_with_dimension(dim),
                    ),
                );
                cache.push((Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first)));
            }
            cache.pop()
        })
    };

    for i in 1..height_of_root {
        let partial_scan = probes[i as usize - 1] < cells[(height_of_root - 1 - i) as usize];
        if partial_scan || is_residual {
            state = step(state).take(probes[i as usize - 1] as _).collect();
        } else {
            // fast path
            let mut results = LinkedVec::new();
            for (Reverse(_), AlwaysEqual(_), AlwaysEqual(first)) in state {
                tape::read_h1_tape::<R, _, _>(
                    by_next(index, first),
                    || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                    |(), _, norm, first, _| {
                        results.push((
                            Reverse(Distance::ZERO),
                            AlwaysEqual(norm),
                            AlwaysEqual(first),
                        ));
                    },
                );
            }
            state = results.into_vec();
        }
    }

    let mut results = LinkedVec::<(_, AlwaysEqual<_>)>::new();
    let candidate_count = Cell::new(0_usize);
    let candidate_count_before_block_prune = Cell::new(0_usize);
    let candidate_count_after_block_prune = Cell::new(0_usize);
    let block_summary_checked_count = Cell::new(0_usize);
    let block_summary_rejected_count = Cell::new(0_usize);
    let block_summary_maybe_count = Cell::new(0_usize);
    let block_summary_candidates_skipped_count = Cell::new(0_usize);
    for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut callback = id_5(
            |(rough, err), head, payload, candidate_metadata, prefetch| {
                candidate_count.set(candidate_count.get() + 1);
                candidate_count_before_block_prune
                    .set(candidate_count_before_block_prune.get() + 1);
                candidate_count_after_block_prune.set(candidate_count_after_block_prune.get() + 1);
                if !candidate_filter(payload, candidate_metadata) {
                    return;
                }
                let lowerbound = Distance::from_f32(rough - err * epsilon);
                results.push((
                    (Reverse(lowerbound), AlwaysEqual(())),
                    AlwaysEqual(PackedRefMut4(bump.alloc((
                        payload,
                        head,
                        candidate_metadata,
                        BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                    )))),
                ));
            },
        );
        if prefetch_h0_tuples.is_not_plain() {
            let directory =
                tape::read_directory_tape::<R>(by_next(index, jump_tuple.directory_first()));
            tape::read_frozen_tape_with_block_prune::<R, _, _>(
                by_directory(&mut prefetch_h0_tuples, directory),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                |candidate_metadata, payload| {
                    let candidate_len = payload.iter().filter(|payload| payload.is_some()).count();
                    match block_prune(candidate_metadata, payload) {
                        BlockPrune::Disabled => BlockPrune::Scan,
                        BlockPrune::Scan => {
                            block_summary_checked_count.set(block_summary_checked_count.get() + 1);
                            block_summary_maybe_count.set(block_summary_maybe_count.get() + 1);
                            BlockPrune::Scan
                        }
                        BlockPrune::Skip => {
                            block_summary_checked_count.set(block_summary_checked_count.get() + 1);
                            block_summary_rejected_count
                                .set(block_summary_rejected_count.get() + 1);
                            block_summary_candidates_skipped_count
                                .set(block_summary_candidates_skipped_count.get() + candidate_len);
                            candidate_count.set(candidate_count.get() + candidate_len);
                            candidate_count_before_block_prune
                                .set(candidate_count_before_block_prune.get() + candidate_len);
                            BlockPrune::Skip
                        }
                    }
                },
                &mut callback,
            );
        } else {
            tape::read_frozen_tape_with_block_prune::<R, _, _>(
                by_next(index, jump_tuple.frozen_first()),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                |candidate_metadata, payload| {
                    let candidate_len = payload.iter().filter(|payload| payload.is_some()).count();
                    match block_prune(candidate_metadata, payload) {
                        BlockPrune::Disabled => BlockPrune::Scan,
                        BlockPrune::Scan => {
                            block_summary_checked_count.set(block_summary_checked_count.get() + 1);
                            block_summary_maybe_count.set(block_summary_maybe_count.get() + 1);
                            BlockPrune::Scan
                        }
                        BlockPrune::Skip => {
                            block_summary_checked_count.set(block_summary_checked_count.get() + 1);
                            block_summary_rejected_count
                                .set(block_summary_rejected_count.get() + 1);
                            block_summary_candidates_skipped_count
                                .set(block_summary_candidates_skipped_count.get() + candidate_len);
                            candidate_count.set(candidate_count.get() + candidate_len);
                            candidate_count_before_block_prune
                                .set(candidate_count_before_block_prune.get() + candidate_len);
                            BlockPrune::Skip
                        }
                    }
                },
                &mut callback,
            );
        }
        tape::read_appendable_tape::<R, _>(
            by_next(index, jump_tuple.appendable_first()),
            O::binary_access(&lut.1, is_residual, dis_f.to_f32(), norm),
            &mut callback,
        );
    }
    (
        results.into_vec(),
        DefaultSearchStats {
            candidate_count: candidate_count.get(),
            candidate_count_before_block_prune: candidate_count_before_block_prune.get(),
            candidate_count_after_block_prune: candidate_count_after_block_prune.get(),
            block_summary_checked_count: block_summary_checked_count.get(),
            block_summary_rejected_count: block_summary_rejected_count.get(),
            block_summary_maybe_count: block_summary_maybe_count.get(),
            block_summary_candidates_skipped_count: block_summary_candidates_skipped_count.get(),
        },
    )
}

pub fn maxsim_search<'b, R: RelationRead, O: Operator>(
    index: &'b R,
    vector: <O::Vector as VectorOwned>::Borrowed<'_>,
    probes: Vec<u32>,
    epsilon: f32,
    mut threshold: u32,
    bump: &'b impl Bump,
    mut prefetch_h1_vectors: impl PrefetcherHeapFamily<'b, R>,
    mut prefetch_h0_tuples: impl PrefetcherSequenceFamily<'b, R>,
) -> (
    Vec<(
        (Reverse<Distance>, AlwaysEqual<Distance>),
        AlwaysEqual<PackedRefMut8<'b, (NonZero<u64>, u16, BorrowedIter<'b>)>>,
    )>,
    Distance,
)
where
    R::Page: Page<Opaque = Opaque>,
{
    let meta_guard = index.read(0);
    let meta_bytes = meta_guard.get(1).expect("data corruption");
    let meta_tuple = MetaTuple::deserialize_ref(meta_bytes);
    let dim = meta_tuple.dim();
    let is_residual = meta_tuple.is_residual();
    let height_of_root = meta_tuple.height_of_root();
    let cells = meta_tuple.cells().to_vec();
    assert_eq!(dim, vector.dim(), "unmatched dimensions");
    if height_of_root as usize != 1 + probes.len() {
        panic!(
            "usage: need {} probes, but {} probes provided",
            height_of_root - 1,
            probes.len()
        );
    }
    debug_assert_eq!(cells[(height_of_root - 1) as usize], 1);

    type State = Vec<(Reverse<Distance>, AlwaysEqual<f32>, AlwaysEqual<u32>)>;
    let mut state: State = if is_residual {
        let prefetch =
            BorrowedIter::from_slice(meta_tuple.centroid_prefetch(), |x| bump.alloc_slice(x));
        let head = meta_tuple.centroid_head();
        let distance = centroids::read::<R, O, _>(
            prefetch.map(|id| index.read(id)),
            head,
            LAccess::new(
                O::Vector::unpack(vector),
                O::DistanceAccessor::default_with_dimension(dim),
            ),
        );
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        vec![(Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))]
    } else {
        // fast path
        let distance = Distance::ZERO;
        let norm = meta_tuple.centroid_norm();
        let first = meta_tuple.first();
        vec![(Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first))]
    };

    drop(meta_guard);
    let lut = O::Vector::preprocess(vector);

    let mut step = |state: State| {
        let mut results = LinkedVec::<(_, AlwaysEqual<Extra1<'b>>)>::new();
        for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
            tape::read_h1_tape::<R, _, _>(
                by_next(index, first),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                |(rough, err), head, norm, first, prefetch| {
                    let lowerbound = Distance::from_f32(rough - err * epsilon);
                    results.push((
                        Reverse(lowerbound),
                        AlwaysEqual(bump.alloc((
                            first,
                            norm,
                            head,
                            BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                        ))),
                    ));
                },
            );
        }
        let mut heap = prefetch_h1_vectors.prefetch(results.into_vec());
        let mut cache = BinaryHeap::<(_, _, _)>::new();
        std::iter::from_fn(move || {
            while let Some(((Reverse(_), AlwaysEqual(&mut (first, norm, head, ..))), prefetch)) =
                heap.next_if(|(d, _)| Some(*d) > cache.peek().map(|(d, ..)| *d))
            {
                let distance = centroids::read::<R, O, _>(
                    prefetch,
                    head,
                    LAccess::new(
                        O::Vector::unpack(vector),
                        O::DistanceAccessor::default_with_dimension(dim),
                    ),
                );
                cache.push((Reverse(distance), AlwaysEqual(norm), AlwaysEqual(first)));
            }
            cache.pop()
        })
    };

    let mut it = None;
    for i in 1..height_of_root {
        let partial_scan = probes[i as usize - 1] < cells[(height_of_root - 1 - i) as usize];
        let needs_sort = i + 1 == height_of_root && threshold != 0;
        if partial_scan || is_residual || needs_sort {
            let it = it.insert(step(state));
            state = it.take(probes[i as usize - 1] as _).collect();
        } else {
            // fast path
            let mut results = LinkedVec::new();
            for (Reverse(_), AlwaysEqual(_), AlwaysEqual(first)) in state {
                tape::read_h1_tape::<R, _, _>(
                    by_next(index, first),
                    || FunctionalAccessor::new((), id_0(|_, _| ()), id_1(|_, _| [(); _])),
                    |(), _, norm, first, _| {
                        results.push((
                            Reverse(Distance::ZERO),
                            AlwaysEqual(norm),
                            AlwaysEqual(first),
                        ));
                    },
                );
            }
            state = results.into_vec();
        }
    }

    let mut results = LinkedVec::<(_, AlwaysEqual<_>)>::new();
    for (Reverse(dis_f), AlwaysEqual(norm), AlwaysEqual(first)) in state {
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        let mut callback = id_5(
            |(rough, err), head, payload, _candidate_metadata, prefetch| {
                let lowerbound = Distance::from_f32(rough - err * epsilon);
                let rough = Distance::from_f32(rough);
                results.push((
                    (Reverse(lowerbound), AlwaysEqual(rough)),
                    AlwaysEqual(PackedRefMut8(bump.alloc((
                        payload,
                        head,
                        BorrowedIter::from_slice(prefetch, |x| bump.alloc_slice(x)),
                    )))),
                ));
            },
        );
        if prefetch_h0_tuples.is_not_plain() {
            let directory =
                tape::read_directory_tape::<R>(by_next(index, jump_tuple.directory_first()));
            tape::read_frozen_tape::<R, _, _>(
                by_directory(&mut prefetch_h0_tuples, directory),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                &mut callback,
            );
        } else {
            tape::read_frozen_tape::<R, _, _>(
                by_next(index, jump_tuple.frozen_first()),
                || O::block_access(&lut.0, is_residual, dis_f.to_f32(), norm),
                &mut callback,
            );
        }
        tape::read_appendable_tape::<R, _>(
            by_next(index, jump_tuple.appendable_first()),
            O::binary_access(&lut.1, is_residual, dis_f.to_f32(), norm),
            &mut callback,
        );
        threshold = threshold.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as _);
    }
    let mut estimation_by_threshold = Distance::NEG_INFINITY;
    for (Reverse(distance), AlwaysEqual(_), AlwaysEqual(first)) in it.into_iter().flatten() {
        if threshold == 0 {
            break;
        }
        let jump_guard = index.read(first);
        let jump_bytes = jump_guard.get(1).expect("data corruption");
        let jump_tuple = JumpTuple::deserialize_ref(jump_bytes);
        threshold = threshold.saturating_sub(jump_tuple.tuples().min(u32::MAX as _) as _);
        estimation_by_threshold = distance;
    }
    (results.into_vec(), estimation_by_threshold)
}
