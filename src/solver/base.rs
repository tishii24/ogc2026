use std::collections::BTreeMap;

use crate::{
    Operation, Problem, ScheduledBlock, Solution, utils::base::rand::Random,
    utils::precompute::Precompute,
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum NeighborKind {
    LargeReconstruct,
    BeamLargeReconstruct,
    Shift,
    Move,
    Rotate,
    Swap,
}

impl NeighborKind {
    pub(crate) fn index(&self) -> usize {
        match self {
            NeighborKind::LargeReconstruct => 0,
            NeighborKind::BeamLargeReconstruct => 1,
            NeighborKind::Shift => 2,
            NeighborKind::Move => 3,
            NeighborKind::Rotate => 4,
            NeighborKind::Swap => 5,
        }
    }
}

pub(crate) fn score13_block(problem: &Problem, pre: &Precompute, s: ScheduledBlock) -> f64 {
    let block = &problem.blocks[s.block_id];
    let tardiness = (s.exit_time - block.due_date).max(0);
    let pref_penalty = pre.pref_penalty[s.block_id][s.bay_id];
    problem.weights.w1 * tardiness as f64 + problem.weights.w3 * pref_penalty as f64
}

pub(crate) fn schedule_tardiness(problem: &Problem, schedule: &[ScheduledBlock]) -> i64 {
    schedule
        .iter()
        .map(|s| (s.exit_time - problem.blocks[s.block_id].due_date).max(0))
        .sum()
}

pub(crate) fn score_schedule(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
) -> (f64, i64) {
    let mut obj1 = 0;
    let mut obj3 = 0.0;
    let mut loads = vec![0.0; problem.bays.len()];

    for s in schedule {
        let block = &problem.blocks[s.block_id];
        obj1 += (s.exit_time - block.due_date).max(0);
        loads[s.bay_id] += block.workload as f64;
        obj3 += pre.pref_penalty[s.block_id][s.bay_id] as f64;
    }

    let obj2 = normalized_imbalance(pre, &loads);
    (
        problem.weights.w1 * obj1 as f64 + problem.weights.w2 * obj2 + problem.weights.w3 * obj3,
        obj1,
    )
}

pub(crate) fn normalized_imbalance(pre: &Precompute, loads: &[f64]) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }

    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let normalized = load * pre.bay_load_scale[bay_id];
        min_value = min_value.min(normalized);
        max_value = max_value.max(normalized);
    }
    (max_value - min_value).floor()
}

pub(crate) fn schedule_to_solution(schedule: &[ScheduledBlock]) -> Solution {
    let mut operations: BTreeMap<i64, Vec<Operation>> = BTreeMap::new();

    for s in schedule {
        operations.entry(s.exit_time).or_default().push(Operation {
            op_type: "EXIT",
            block_id: s.block_id,
            bay_id: s.bay_id,
            x: None,
            y: None,
            orient_idx: None,
        });
    }
    for s in schedule {
        operations.entry(s.entry_time).or_default().push(Operation {
            op_type: "ENTRY",
            block_id: s.block_id,
            bay_id: s.bay_id,
            x: Some(s.x),
            y: Some(s.y),
            orient_idx: Some(s.orient_idx),
        });
    }

    operations.retain(|_, ops| !ops.is_empty());
    Solution { operations }
}

pub(crate) fn sample_neighbor<R: Random>(
    rng: &mut R,
    probs: &[(NeighborKind, f64)],
) -> NeighborKind {
    let total = probs.iter().map(|&(_, prob)| prob).sum::<f64>();
    debug_assert!(total > 0.0);

    let mut x = rng.nextf() * total;
    for &(kind, prob) in probs {
        if x < prob {
            return kind;
        }
        x -= prob;
    }
    probs.iter().rev().find(|&&(_, prob)| prob > 0.0).unwrap().0
}

pub(crate) fn gen_rangef(rng: &mut impl Random, r: (f64, f64)) -> f64 {
    rng.gen_rangef(r.0, r.1)
}
