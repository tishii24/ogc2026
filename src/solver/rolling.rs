use std::ops::Range;

use crate::{
    INF, Problem, ScheduledBlock,
    params::{AnnealingParamsConfig, InsertParams, NeighborParams, RollingPhaseParams},
    utils::{random::RandPcg64Mcg, time::Timer},
};

use super::{
    PreoptimizeState,
    insert::insert_greedy,
    optimize::{OptimizeAnnealing, OptimizeMode, OptimizeState, make_optimize_state},
    precompute::Precompute,
};

const EPS: f64 = 1e-9;

fn build_admission_order(problem: &Problem, preopt: &PreoptimizeState) -> Vec<usize> {
    let mut order: Vec<_> = (0..problem.blocks.len()).collect();
    order.sort_by_key(|&block_id| {
        (
            preopt.blocks[block_id].entry_time,
            problem.blocks[block_id].due_date,
            block_id,
        )
    });
    order
}

fn build_horizons(
    order: &[usize],
    preopt: &PreoptimizeState,
    horizon_size: usize,
) -> Vec<Range<usize>> {
    let mut horizons = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let mut end = (start + horizon_size).min(order.len());
        while end < order.len()
            && preopt.blocks[order[end - 1]].entry_time == preopt.blocks[order[end]].entry_time
        {
            end += 1;
        }
        horizons.push(start..end);
        start = end;
    }
    horizons
}

fn extend_schedule(
    problem: &Problem,
    pre: &Precompute,
    preopt: &PreoptimizeState,
    state: OptimizeState,
    added_block_ids: &[usize],
    insert_params: &InsertParams,
    rng: &mut RandPcg64Mcg,
) -> OptimizeState {
    let mut schedule = state.schedule;
    let mut loads = vec![0.0; problem.bays.len()];
    for scheduled in &schedule {
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
    }

    for &block_id in added_block_ids {
        let block = &problem.blocks[block_id];
        let entry_time = preopt.blocks[block_id].entry_time.max(block.release_time);
        let original = ScheduledBlock {
            block_id,
            bay_id: preopt.blocks[block_id].bay_id,
            orient_idx: 0,
            x: 0,
            y: 0,
            entry_time,
            exit_time: entry_time + block.processing_time,
        };
        loop {
            if let Some(scheduled) = insert_greedy(
                problem,
                pre,
                original,
                block.release_time,
                INF,
                &schedule,
                &loads,
                insert_params,
                &pre.bay_order_by_pref[block_id],
                1,
                1.0,
                0.0,
                rng,
            ) {
                loads[scheduled.bay_id] += block.workload as f64;
                schedule.push(scheduled);
                break;
            }
        }
    }

    make_optimize_state(problem, pre, schedule, 0.0)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_rolling_initial_state(
    problem: &Problem,
    pre: &Precompute,
    preopt: &PreoptimizeState,
    deadline: f64,
    timer: Timer,
    phase_params: &RollingPhaseParams,
    annealing_params: &AnnealingParamsConfig,
    neighbor_params: &NeighborParams,
    insert_params: &InsertParams,
    max_worker_count: usize,
    seed: u64,
) -> OptimizeState {
    let order = build_admission_order(problem, preopt);
    let horizons = build_horizons(&order, preopt, phase_params.horizon_size);
    let mut state = make_optimize_state(problem, pre, Vec::new(), 0.0);
    let mut rng = RandPcg64Mcg::new(seed);
    let annealer = OptimizeAnnealing::new(problem, pre, timer);

    for (horizon_index, horizon) in horizons.iter().enumerate() {
        state = extend_schedule(
            problem,
            pre,
            preopt,
            state,
            &order[horizon.clone()],
            insert_params,
            &mut rng,
        );
        log!(
            "[{:.4}] [rolling] horizon={}/{}, blocks={}, initial={:.3}",
            timer.elapsed_seconds(),
            horizon_index + 1,
            horizons.len(),
            horizon.end,
            state.objective,
        );

        if state.objective <= EPS {
            continue;
        }
        let now = timer.elapsed_seconds();
        if now >= deadline {
            continue;
        }
        let remaining_weight: f64 = horizons[horizon_index..]
            .iter()
            .map(|remaining| (remaining.end as f64).powf(phase_params.time_allocation_power))
            .sum();
        let current_weight = (horizon.end as f64).powf(phase_params.time_allocation_power);
        let horizon_deadline = now + (deadline - now) * current_weight / remaining_weight;
        state = annealer.run(
            state,
            horizon_deadline,
            annealing_params,
            neighbor_params,
            insert_params,
            OptimizeMode {
                name: "rolling",
                w2: 0.0,
            },
            None,
            seed.wrapping_add(((horizon_index + 1) as u64) << 32),
            max_worker_count,
        );
    }

    make_optimize_state(problem, pre, state.schedule, problem.weights.w2)
}
