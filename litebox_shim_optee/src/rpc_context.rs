// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! RPC context tracking for multi-call OP-TEE operations.

use alloc::boxed::Box;
use core::sync::atomic::{AtomicU32, Ordering};
use hashbrown::HashMap;
use once_cell::race::OnceBox;
use spin::mutex::SpinMutex;

/// Maximum number of RPC contexts that may be active at once.
pub const MAX_RPC_CONTEXTS: usize = 1024;

/// Progress of an RPC-backed Dynamic TA request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpcStage {
    LoadTaSize,
    ShmAlloc,
    LoadTaBinary,
    ShmFree,
}

/// An RPC context map operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpcContextError {
    Full,
    AlreadyExists,
    NotFound,
    UnexpectedStage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RpcContext {
    stage: RpcStage,
    regd_shm_offset: usize,
}

/// Maps RPC context IDs to trusted continuation state.
pub struct RpcContextMap {
    next_id: AtomicU32,
    inner: SpinMutex<HashMap<u32, RpcContext>>,
    max_contexts: usize,
}

impl RpcContextMap {
    /// Create an empty RPC context map.
    pub fn new() -> Self {
        Self::with_limits(0, MAX_RPC_CONTEXTS)
    }

    fn with_limits(next_id: u32, max_contexts: usize) -> Self {
        Self {
            next_id: AtomicU32::new(next_id),
            inner: SpinMutex::new(HashMap::new()),
            max_contexts,
        }
    }

    /// Allocate a unique context ID and associate it with trusted continuation state.
    pub fn allocate(
        &self,
        stage: RpcStage,
        regd_shm_offset: usize,
    ) -> Result<u32, RpcContextError> {
        let mut contexts = self.inner.lock();
        if contexts.len() >= self.max_contexts {
            return Err(RpcContextError::Full);
        }

        // With N active entries, at least one of N + 1 consecutive IDs is free.
        for _ in 0..=contexts.len() {
            let context_id = self.next_id.fetch_add(1, Ordering::Relaxed);
            if let hashbrown::hash_map::Entry::Vacant(entry) = contexts.entry(context_id) {
                entry.insert(RpcContext {
                    stage,
                    regd_shm_offset,
                });
                return Ok(context_id);
            }
        }

        Err(RpcContextError::Full)
    }

    /// Insert a context with a caller-provided ID.
    pub fn insert(&self, context_id: u32, stage: RpcStage) -> Result<(), RpcContextError> {
        let mut contexts = self.inner.lock();
        if contexts.contains_key(&context_id) {
            return Err(RpcContextError::AlreadyExists);
        }
        if contexts.len() >= self.max_contexts {
            return Err(RpcContextError::Full);
        }
        contexts.insert(
            context_id,
            RpcContext {
                stage,
                regd_shm_offset: 0,
            },
        );
        Ok(())
    }

    /// Get the current stage for `context_id`.
    pub fn get_curr_stage(&self, context_id: u32) -> Option<RpcStage> {
        self.inner
            .lock()
            .get(&context_id)
            .map(|context| context.stage)
    }

    /// Get the registered-SHM offset captured before `args[3]` became the context ID.
    pub fn get_regd_shm_offset(&self, context_id: u32) -> Option<usize> {
        self.inner
            .lock()
            .get(&context_id)
            .map(|context| context.regd_shm_offset)
    }

    /// Atomically advance a context from `expected` to `next`.
    pub fn transition(
        &self,
        context_id: u32,
        expected: RpcStage,
        next: RpcStage,
    ) -> Result<(), RpcContextError> {
        let mut contexts = self.inner.lock();
        let context = contexts
            .get_mut(&context_id)
            .ok_or(RpcContextError::NotFound)?;
        if context.stage != expected {
            return Err(RpcContextError::UnexpectedStage);
        }
        context.stage = next;
        Ok(())
    }

    /// Remove and return a context's stage.
    pub fn take(&self, context_id: u32) -> Option<RpcStage> {
        self.inner
            .lock()
            .remove(&context_id)
            .map(|context| context.stage)
    }

    /// Return the number of active RPC contexts.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Return whether there are no active RPC contexts.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }
}

impl Default for RpcContextMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Return the process-wide RPC context map.
pub fn rpc_context_map() -> &'static RpcContextMap {
    static RPC_CONTEXT_MAP: OnceBox<RpcContextMap> = OnceBox::new();
    RPC_CONTEXT_MAP.get_or_init(|| Box::new(RpcContextMap::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_unique_context_ids() {
        let contexts = RpcContextMap::new();
        let first = contexts.allocate(RpcStage::LoadTaSize, 0x100).unwrap();
        let second = contexts.allocate(RpcStage::LoadTaSize, 0x200).unwrap();

        assert_ne!(first, second);
        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts.get_regd_shm_offset(first), Some(0x100));
        assert_eq!(contexts.get_regd_shm_offset(second), Some(0x200));
    }

    #[test]
    fn transitions_a_stable_context_id() {
        let contexts = RpcContextMap::new();
        let context_id = contexts.allocate(RpcStage::LoadTaSize, 0).unwrap();

        assert_eq!(
            contexts.transition(context_id, RpcStage::LoadTaSize, RpcStage::ShmAlloc),
            Ok(())
        );
        assert_eq!(
            contexts.get_curr_stage(context_id),
            Some(RpcStage::ShmAlloc)
        );
        assert_eq!(
            contexts.transition(context_id, RpcStage::LoadTaSize, RpcStage::LoadTaBinary),
            Err(RpcContextError::UnexpectedStage)
        );
    }

    #[test]
    fn taking_a_context_rejects_replay() {
        let contexts = RpcContextMap::new();
        let context_id = contexts.allocate(RpcStage::ShmFree, 0).unwrap();

        assert_eq!(contexts.take(context_id), Some(RpcStage::ShmFree));
        assert_eq!(contexts.take(context_id), None);
        assert!(contexts.is_empty());
    }

    #[test]
    fn enforces_capacity_and_duplicate_ids() {
        let contexts = RpcContextMap::with_limits(0, 1);
        assert_eq!(contexts.insert(7, RpcStage::LoadTaSize), Ok(()));
        assert_eq!(
            contexts.insert(7, RpcStage::ShmAlloc),
            Err(RpcContextError::AlreadyExists)
        );
        assert_eq!(
            contexts.allocate(RpcStage::LoadTaSize, 0),
            Err(RpcContextError::Full)
        );
    }

    #[test]
    fn allocation_wraps_and_skips_active_ids() {
        let contexts = RpcContextMap::with_limits(u32::MAX, 3);
        contexts.insert(u32::MAX, RpcStage::LoadTaSize).unwrap();

        let context_id = contexts.allocate(RpcStage::ShmAlloc, 0).unwrap();
        assert_eq!(context_id, 0);
        assert_eq!(
            contexts.get_curr_stage(u32::MAX),
            Some(RpcStage::LoadTaSize)
        );
        assert_eq!(contexts.get_curr_stage(0), Some(RpcStage::ShmAlloc));
    }
}
