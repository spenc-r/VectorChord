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

use index::prefetcher::Sequence;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub trait HeapBlockKey: Copy {
    fn heap_block_key(self) -> [u16; 3];
}

impl HeapBlockKey for [u16; 3] {
    fn heap_block_key(self) -> [u16; 3] {
        self
    }
}

pub struct Filter<S, P> {
    sequence: S,
    predicate: P,
}

impl<S, P> Sequence for Filter<S, P>
where
    S: Sequence,
    P: FnMut(&S::Item) -> bool,
{
    type Item = S::Item;

    type Inner = S::Inner;

    fn next(&mut self) -> Option<Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
        }
        self.sequence.next()
    }

    fn peek(&mut self) -> Option<&Self::Item> {
        while !(self.predicate)(self.sequence.peek()?) {
            let _ = self.sequence.next();
        }
        self.sequence.peek()
    }

    fn into_inner(self) -> Self::Inner {
        self.sequence.into_inner()
    }
}

pub fn filter<S, P>(sequence: S, predicate: P) -> Filter<S, P> {
    Filter {
        sequence,
        predicate,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WindowFilterStats {
    pub duration: Duration,
    pub candidates: usize,
    pub heap_blocks: usize,
    pub passed: usize,
}

pub struct WindowFilter<S, K, P, O, A>
where
    S: Sequence,
{
    sequence: S,
    key: K,
    predicate: P,
    observer: O,
    window_size: usize,
    ready: VecDeque<S::Item>,
    _phantom: std::marker::PhantomData<fn(A)>,
}

impl<S, K, P, O, A> WindowFilter<S, K, P, O, A>
where
    S: Sequence,
    K: FnMut(&S::Item) -> A,
    P: FnMut(A) -> bool,
    O: FnMut(WindowFilterStats),
    A: HeapBlockKey,
{
    fn fill(&mut self) {
        if self.window_size == 0 {
            while self.ready.is_empty() {
                let Some(item) = self.sequence.next() else {
                    return;
                };
                let key = (self.key)(&item);
                if (self.predicate)(key) {
                    self.ready.push_back(item);
                }
            }
            return;
        }
        while self.ready.is_empty() {
            let started_at = Instant::now();
            let mut window = Vec::new();
            for ordinal in 0..self.window_size {
                let Some(item) = self.sequence.next() else {
                    break;
                };
                let key = (self.key)(&item);
                window.push((key, ordinal, item));
            }
            if window.is_empty() {
                return;
            }
            let candidates = window.len();
            let mut heap_blocks = Vec::with_capacity(candidates);
            for (key, ..) in &window {
                let key = key.heap_block_key();
                heap_blocks.push((key[0], key[1]));
            }
            heap_blocks.sort_unstable();
            heap_blocks.dedup();
            window.sort_by_key(|(key, _, _)| {
                let key = key.heap_block_key();
                (key[0], key[1], key[2])
            });
            let mut passed = Vec::new();
            for (key, ordinal, item) in window {
                if (self.predicate)(key) {
                    passed.push((ordinal, item));
                }
            }
            let passed_count = passed.len();
            passed.sort_by_key(|(ordinal, _)| *ordinal);
            (self.observer)(WindowFilterStats {
                duration: started_at.elapsed(),
                candidates,
                heap_blocks: heap_blocks.len(),
                passed: passed_count,
            });
            self.ready.extend(passed.into_iter().map(|(_, item)| item));
        }
    }
}

impl<S, K, P, O, A> Sequence for WindowFilter<S, K, P, O, A>
where
    S: Sequence,
    K: FnMut(&S::Item) -> A,
    P: FnMut(A) -> bool,
    O: FnMut(WindowFilterStats),
    A: HeapBlockKey,
{
    type Item = S::Item;

    type Inner = std::iter::Chain<std::collections::vec_deque::IntoIter<S::Item>, S::Inner>;

    fn next(&mut self) -> Option<Self::Item> {
        self.fill();
        self.ready.pop_front()
    }

    fn peek(&mut self) -> Option<&Self::Item> {
        self.fill();
        self.ready.front()
    }

    fn into_inner(self) -> Self::Inner {
        self.ready.into_iter().chain(self.sequence.into_inner())
    }
}

pub fn window_filter<S, K, P, O, A>(
    sequence: S,
    window_size: usize,
    key: K,
    predicate: P,
    observer: O,
) -> WindowFilter<S, K, P, O, A>
where
    S: Sequence,
    K: FnMut(&S::Item) -> A,
    P: FnMut(A) -> bool,
    O: FnMut(WindowFilterStats),
    A: HeapBlockKey,
{
    WindowFilter {
        sequence,
        key,
        predicate,
        observer,
        window_size,
        ready: VecDeque::new(),
        _phantom: std::marker::PhantomData,
    }
}
