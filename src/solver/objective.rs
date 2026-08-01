use crate::{Problem, ScheduledBlock};

use super::precompute::Precompute;

#[derive(Clone, Copy, Debug)]
pub(super) struct RawScore {
    pub(super) z1: i64,
    pub(super) z2: f64,
    pub(super) z3: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ScoreWeights {
    pub(super) w1: f64,
    pub(super) w2: f64,
    pub(super) w3: f64,
}

impl ScoreWeights {
    pub(super) fn official(problem: &Problem) -> Self {
        Self {
            w1: problem.weights.w1,
            w2: problem.weights.w2,
            w3: problem.weights.w3,
        }
    }

    pub(super) fn tardiness_only(problem: &Problem) -> Self {
        Self {
            w1: problem.weights.w1,
            w2: 0.0,
            w3: 0.0,
        }
    }
}

impl RawScore {
    pub(super) fn weighted(self, weights: ScoreWeights) -> f64 {
        weights.w1 * self.z1 as f64 + weights.w2 * self.z2 + weights.w3 * self.z3
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ScheduleScore {
    pub(super) raw: RawScore,
    pub(super) objective: f64,
}

pub(super) fn score13_block_weighted(
    problem: &Problem,
    pre: &Precompute,
    scheduled: ScheduledBlock,
    weights: ScoreWeights,
) -> f64 {
    let block = &problem.blocks[scheduled.block_id];
    let tardiness = (scheduled.exit_time - block.due_date).max(0);
    let pref_penalty = pre.pref_penalty[scheduled.block_id][scheduled.bay_id];
    weights.w1 * tardiness as f64 + weights.w3 * pref_penalty as f64
}

pub(super) fn schedule_tardiness(problem: &Problem, schedule: &[ScheduledBlock]) -> i64 {
    schedule
        .iter()
        .map(|scheduled| (scheduled.exit_time - problem.blocks[scheduled.block_id].due_date).max(0))
        .sum()
}

pub(super) fn score_schedule(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
) -> ScheduleScore {
    let mut obj1 = 0;
    let mut obj3 = 0.0;
    let mut loads = vec![0.0; problem.bays.len()];

    for scheduled in schedule {
        let block = &problem.blocks[scheduled.block_id];
        obj1 += (scheduled.exit_time - block.due_date).max(0);
        loads[scheduled.bay_id] += block.workload as f64;
        obj3 += pre.pref_penalty[scheduled.block_id][scheduled.bay_id] as f64;
    }

    let raw = RawScore {
        z1: obj1,
        z2: normalized_imbalance(&loads, &pre.bay_load_scale),
        z3: obj3,
    };
    ScheduleScore {
        objective: raw.weighted(ScoreWeights::official(problem)),
        raw,
    }
}

pub(super) fn normalized_imbalance(loads: &[f64], bay_load_scale: &[f64]) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }

    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let normalized = load * bay_load_scale[bay_id];
        min_value = min_value.min(normalized);
        max_value = max_value.max(normalized);
    }
    (max_value - min_value).floor()
}
