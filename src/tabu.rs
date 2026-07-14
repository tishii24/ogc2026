use std::collections::{HashSet, VecDeque};

use crate::ScheduledBlock;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ScheduleKey(u64);

pub(crate) struct ScheduleTabu {
    capacity: usize,
    queue: VecDeque<ScheduleKey>,
    set: HashSet<ScheduleKey>,
}

impl ScheduleTabu {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            queue: VecDeque::with_capacity(capacity),
            set: HashSet::with_capacity(capacity * 2),
        }
    }

    pub(crate) fn key(schedule: &[ScheduledBlock]) -> ScheduleKey {
        let mut hash = mix_hash(1469598103934665603, schedule.len() as u64);
        for &block in schedule {
            hash ^= hash_scheduled_block(block);
        }
        ScheduleKey(hash)
    }

    pub(crate) fn contains(&self, key: ScheduleKey) -> bool {
        self.set.contains(&key)
    }

    pub(crate) fn insert(&mut self, key: ScheduleKey) {
        if !self.set.insert(key) {
            return;
        }
        self.queue.push_back(key);
        if self.queue.len() > self.capacity {
            if let Some(old_key) = self.queue.pop_front() {
                self.set.remove(&old_key);
            }
        }
    }

    pub(crate) fn insert_schedule(&mut self, schedule: &[ScheduledBlock]) {
        self.insert(Self::key(schedule));
    }
}

fn mix_hash(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash.wrapping_mul(1099511628211)
}

fn hash_scheduled_block(block: ScheduledBlock) -> u64 {
    let mut hash = 1469598103934665603;
    hash = mix_hash(hash, block.block_id as u64);
    hash = mix_hash(hash, block.bay_id as u64);
    hash = mix_hash(hash, block.orient_idx as u64);
    hash = mix_hash(hash, block.x as u64);
    hash = mix_hash(hash, block.y as u64);
    hash = mix_hash(hash, block.entry_time as u64);
    mix_hash(hash, block.exit_time as u64)
}
