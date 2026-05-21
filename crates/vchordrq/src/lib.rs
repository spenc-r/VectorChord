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

mod build;
mod bulkdelete;
mod cache;
mod centroids;
mod closure_lifetime_binder;
mod consume;
mod cost;
mod fast_heap;
mod freepages;
mod insert;
mod linked_vec;
mod maintain;
mod prewarm;
mod rerank;
mod search;
mod tape;
mod tape_writer;
mod tuples;
mod vectors;

pub mod operator;
pub mod types;

pub use build::build;
pub use bulkdelete::{bulkdelete, bulkdelete_vectors};
pub use cache::cache;
pub use consume::consume;
pub use cost::cost;
pub use fast_heap::FastHeap;
pub use insert::{InsertChooser, insert, insert_vector};
pub use maintain::{MaintainChooser, maintain};
pub use prewarm::prewarm;
pub use rerank::{
    RerankInstrumentation, RerankInstrumentationSnapshot, how, rerank_heap,
    rerank_heap_instrumented, rerank_index, rerank_index_instrumented,
};
pub use search::{
    BlockPrune, DefaultSearchStats, default_search, default_search_with_candidate_filter,
    default_search_with_candidate_filter_and_block_prune, maxsim_search,
};

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct Opaque {
    pub next: u32,
    pub skip: u32,
}

#[allow(unsafe_code)]
unsafe impl index::relation::Opaque for Opaque {}

pub(crate) struct Branch<T> {
    pub code: rabitq::bit::Code,
    pub delta: f32,
    pub prefetch: Vec<u32>,
    pub head: u16,
    pub norm: f32,
    pub extra: T,
    pub candidate_metadata: CandidateMetadata,
}

#[derive(Debug, Clone, Copy)]
pub enum RerankMethod {
    Index,
    Heap,
}

pub const MAX_METADATA_ATTRS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CandidateMetadata {
    valid: u32,
    values: [i64; MAX_METADATA_ATTRS],
}

impl Default for CandidateMetadata {
    fn default() -> Self {
        Self {
            valid: 0,
            values: [0; MAX_METADATA_ATTRS],
        }
    }
}

impl CandidateMetadata {
    pub fn get(self, index: usize) -> Option<i64> {
        if index >= MAX_METADATA_ATTRS || self.valid & (1_u32 << index) == 0 {
            None
        } else {
            Some(self.values[index])
        }
    }

    pub fn set(&mut self, index: usize, value: i64) {
        assert!(index < MAX_METADATA_ATTRS);
        self.valid |= 1_u32 << index;
        self.values[index] = value;
    }

    pub fn valid(self) -> u32 {
        self.valid
    }

    pub fn attr_count(self) -> usize {
        if self.valid == 0 {
            0
        } else {
            (u32::BITS - self.valid.leading_zeros()) as usize
        }
    }

    pub fn is_empty(self) -> bool {
        self.valid == 0
    }
}
