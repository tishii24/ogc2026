use crate::{Problem, ScheduledBlock};

use super::precompute::Precompute;

#[derive(Clone, Copy, Debug)]
pub(super) struct ScheduleScore {
    pub(super) objective: f64,
    pub(super) total_tardiness: i64,
}

pub(super) fn score13_block(problem: &Problem, pre: &Precompute, scheduled: ScheduledBlock) -> f64 {
    let block = &problem.blocks[scheduled.block_id];
    let tardiness = (scheduled.exit_time - block.due_date).max(0);
    let pref_penalty = pre.pref_penalty[scheduled.block_id][scheduled.bay_id];
    problem.weights.w1 * tardiness as f64 + problem.weights.w3 * pref_penalty as f64
}

pub(super) fn score_schedule(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    w2: f64,
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

    let obj2 = score_z2(&loads, &pre.bay_load_scale);
    ScheduleScore {
        objective: problem.weights.w1 * obj1 as f64 + w2 * obj2 + problem.weights.w3 * obj3,
        total_tardiness: obj1,
    }
}

pub(super) fn score_z2(loads: &[f64], bay_load_scale: &[f64]) -> f64 {
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
