use std::time::Instant;

use crate::{
    Problem,
    preoptimize::{
        PreoptimizeData, PreoptimizeParams, PreoptimizeResult, PreoptimizeStatus,
        PreoptimizedBlock, evaluate_schedule, prepare_preoptimize,
    },
    util::rand::{RandPcg64Mcg, Random},
};

const RNG_SEED: u64 = 2;
const SWAP_PROBABILITY: f64 = 0.15;
const BAD_BLOCK_SAMPLE_COUNT: usize = 8;
const BAD_BLOCK_SELECT_PROBABILITY: f64 = 0.75;
const MAX_RELOCATE_ATTEMPTS: usize = 8;
const MAX_TIME_SHIFT: i64 = 10;
const END_TEMPERATURE_RATIO: f64 = 1e-4;

struct AnnealingState {
    schedule: Vec<PreoptimizedBlock>,
    used_area: Vec<Vec<f64>>,
    loads: Vec<f64>,
    z1: f64,
    z2: f64,
    z3: f64,
    objective: f64,
}

fn normalized_imbalance(loads: &[f64], scales: &[f64]) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }
    let mut min_load = f64::INFINITY;
    let mut max_load = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let normalized = scales[bay_id] * load;
        min_load = min_load.min(normalized);
        max_load = max_load.max(normalized);
    }
    max_load - min_load
}

fn block_z1(problem: &Problem, block_id: usize, selected: PreoptimizedBlock) -> f64 {
    let block = &problem.blocks[block_id];
    (selected.entry_time + block.processing_time - block.due_date).max(0) as f64
}

fn block_z3(data: &PreoptimizeData, block_id: usize, selected: PreoptimizedBlock) -> f64 {
    data.pref_penalty[block_id][selected.bay_id] as f64
}

fn add_used_area(
    problem: &Problem,
    data: &PreoptimizeData,
    used_area: &mut [Vec<f64>],
    block_id: usize,
    selected: PreoptimizedBlock,
    sign: f64,
) {
    let area = data.occupancy[block_id][selected.bay_id].unwrap();
    let block = &problem.blocks[block_id];
    for time in selected.entry_time..selected.entry_time + block.processing_time {
        used_area[selected.bay_id][(time - data.min_time) as usize] += sign * area;
    }
}

fn can_place(
    problem: &Problem,
    data: &PreoptimizeData,
    used_area: &[Vec<f64>],
    block_id: usize,
    selected: PreoptimizedBlock,
) -> bool {
    let block = &problem.blocks[block_id];
    if selected.entry_time < block.release_time
        || selected.entry_time + block.processing_time > data.horizon
    {
        return false;
    }
    let Some(area) = data.occupancy[block_id][selected.bay_id] else {
        return false;
    };
    (selected.entry_time..selected.entry_time + block.processing_time).all(|time| {
        used_area[selected.bay_id][(time - data.min_time) as usize] + area
            <= data.bay_areas[selected.bay_id] + 1e-9
    })
}

fn build_state(problem: &Problem, data: &PreoptimizeData) -> AnnealingState {
    let time_count = (data.horizon - data.min_time) as usize;
    let mut used_area = vec![vec![0.0; time_count]; problem.bays.len()];
    let mut loads = vec![0.0; problem.bays.len()];
    for (block_id, &selected) in data.initial.iter().enumerate() {
        add_used_area(problem, data, &mut used_area, block_id, selected, 1.0);
        loads[selected.bay_id] += problem.blocks[block_id].workload as f64;
    }
    let (_, z1, z2, z3) = evaluate_schedule(
        problem,
        &data.pref_penalty,
        &data.bay_load_scale,
        &data.initial,
    );
    AnnealingState {
        schedule: data.initial.clone(),
        used_area,
        loads,
        z1,
        z2,
        z3,
        objective: data.initial_objective,
    }
}

fn select_block(
    problem: &Problem,
    data: &PreoptimizeData,
    state: &AnnealingState,
    rng: &mut impl Random,
) -> usize {
    if rng.nextf() >= BAD_BLOCK_SELECT_PROBABILITY {
        return rng.gen_index(problem.blocks.len());
    }
    let mut best = rng.gen_index(problem.blocks.len());
    let mut best_cost = f64::NEG_INFINITY;
    for _ in 0..BAD_BLOCK_SAMPLE_COUNT {
        let block_id = rng.gen_index(problem.blocks.len());
        let selected = state.schedule[block_id];
        let cost = problem.weights.w1 * block_z1(problem, block_id, selected)
            + problem.weights.w3 * block_z3(data, block_id, selected);
        if cost > best_cost {
            best = block_id;
            best_cost = cost;
        }
    }
    best
}

fn sample_entry_time(
    problem: &Problem,
    data: &PreoptimizeData,
    block_id: usize,
    current: i64,
    rng: &mut impl Random,
) -> i64 {
    let block = &problem.blocks[block_id];
    let min_time = block.release_time;
    let max_time = data.horizon - block.processing_time;
    match rng.gen_range(0, 4) {
        0 => min_time,
        1 => (block.due_date - block.processing_time).clamp(min_time, max_time),
        2 => {
            let delta = rng.gen_range(0, (2 * MAX_TIME_SHIFT + 1) as usize) as i64 - MAX_TIME_SHIFT;
            (current + delta).clamp(min_time, max_time)
        }
        _ => min_time + rng.gen_range(0, (max_time - min_time + 1) as usize) as i64,
    }
}

fn accept(delta: f64, temperature: f64, rng: &mut impl Random) -> bool {
    delta <= 0.0 || rng.nextf() < (-delta / temperature).exp()
}

fn try_relocate(
    problem: &Problem,
    data: &PreoptimizeData,
    state: &mut AnnealingState,
    temperature: f64,
    rng: &mut impl Random,
) -> bool {
    let block_id = select_block(problem, data, state, rng);
    let old = state.schedule[block_id];
    add_used_area(problem, data, &mut state.used_area, block_id, old, -1.0);

    let mut candidate = None;
    for _ in 0..MAX_RELOCATE_ATTEMPTS {
        let bay_id = rng.gen_index(problem.bays.len());
        let entry_time = sample_entry_time(problem, data, block_id, old.entry_time, rng);
        let selected = PreoptimizedBlock { bay_id, entry_time };
        if selected != old && can_place(problem, data, &state.used_area, block_id, selected) {
            candidate = Some(selected);
            break;
        }
    }
    let Some(selected) = candidate else {
        add_used_area(problem, data, &mut state.used_area, block_id, old, 1.0);
        return false;
    };

    let old_z1 = block_z1(problem, block_id, old);
    let new_z1 = block_z1(problem, block_id, selected);
    let old_z3 = block_z3(data, block_id, old);
    let new_z3 = block_z3(data, block_id, selected);
    let mut next_loads = state.loads.clone();
    let workload = problem.blocks[block_id].workload as f64;
    next_loads[old.bay_id] -= workload;
    next_loads[selected.bay_id] += workload;
    let new_z2 = normalized_imbalance(&next_loads, &data.bay_load_scale);
    let delta = problem.weights.w1 * (new_z1 - old_z1)
        + problem.weights.w2 * (new_z2 - state.z2)
        + problem.weights.w3 * (new_z3 - old_z3);

    if accept(delta, temperature, rng) {
        add_used_area(problem, data, &mut state.used_area, block_id, selected, 1.0);
        state.schedule[block_id] = selected;
        state.loads = next_loads;
        state.z1 += new_z1 - old_z1;
        state.z2 = new_z2;
        state.z3 += new_z3 - old_z3;
        state.objective += delta;
        true
    } else {
        add_used_area(problem, data, &mut state.used_area, block_id, old, 1.0);
        false
    }
}

fn try_swap(
    problem: &Problem,
    data: &PreoptimizeData,
    state: &mut AnnealingState,
    temperature: f64,
    rng: &mut impl Random,
) -> bool {
    if problem.blocks.len() < 2 {
        return false;
    }
    let a = select_block(problem, data, state, rng);
    let mut b = rng.gen_index(problem.blocks.len() - 1);
    if b >= a {
        b += 1;
    }
    let old_a = state.schedule[a];
    let old_b = state.schedule[b];
    let (entry_a, entry_b) = if rng.nextf() < 0.5 {
        (old_a.entry_time, old_b.entry_time)
    } else {
        (old_b.entry_time, old_a.entry_time)
    };
    let new_a = PreoptimizedBlock {
        bay_id: old_b.bay_id,
        entry_time: entry_a,
    };
    let new_b = PreoptimizedBlock {
        bay_id: old_a.bay_id,
        entry_time: entry_b,
    };
    if new_a == old_a && new_b == old_b {
        return false;
    }

    add_used_area(problem, data, &mut state.used_area, a, old_a, -1.0);
    add_used_area(problem, data, &mut state.used_area, b, old_b, -1.0);
    if !can_place(problem, data, &state.used_area, a, new_a) {
        add_used_area(problem, data, &mut state.used_area, a, old_a, 1.0);
        add_used_area(problem, data, &mut state.used_area, b, old_b, 1.0);
        return false;
    }
    add_used_area(problem, data, &mut state.used_area, a, new_a, 1.0);
    if !can_place(problem, data, &state.used_area, b, new_b) {
        add_used_area(problem, data, &mut state.used_area, a, new_a, -1.0);
        add_used_area(problem, data, &mut state.used_area, a, old_a, 1.0);
        add_used_area(problem, data, &mut state.used_area, b, old_b, 1.0);
        return false;
    }
    add_used_area(problem, data, &mut state.used_area, b, new_b, 1.0);

    let old_z1 = block_z1(problem, a, old_a) + block_z1(problem, b, old_b);
    let new_z1 = block_z1(problem, a, new_a) + block_z1(problem, b, new_b);
    let old_z3 = block_z3(data, a, old_a) + block_z3(data, b, old_b);
    let new_z3 = block_z3(data, a, new_a) + block_z3(data, b, new_b);
    let mut next_loads = state.loads.clone();
    next_loads[old_a.bay_id] -= problem.blocks[a].workload as f64;
    next_loads[old_b.bay_id] -= problem.blocks[b].workload as f64;
    next_loads[new_a.bay_id] += problem.blocks[a].workload as f64;
    next_loads[new_b.bay_id] += problem.blocks[b].workload as f64;
    let new_z2 = normalized_imbalance(&next_loads, &data.bay_load_scale);
    let delta = problem.weights.w1 * (new_z1 - old_z1)
        + problem.weights.w2 * (new_z2 - state.z2)
        + problem.weights.w3 * (new_z3 - old_z3);

    if accept(delta, temperature, rng) {
        state.schedule[a] = new_a;
        state.schedule[b] = new_b;
        state.loads = next_loads;
        state.z1 += new_z1 - old_z1;
        state.z2 = new_z2;
        state.z3 += new_z3 - old_z3;
        state.objective += delta;
        true
    } else {
        add_used_area(problem, data, &mut state.used_area, a, new_a, -1.0);
        add_used_area(problem, data, &mut state.used_area, b, new_b, -1.0);
        add_used_area(problem, data, &mut state.used_area, a, old_a, 1.0);
        add_used_area(problem, data, &mut state.used_area, b, old_b, 1.0);
        false
    }
}

pub fn preoptimize_annealing(
    problem: &Problem,
    params: PreoptimizeParams,
) -> Result<PreoptimizeResult, String> {
    let build_start = Instant::now();
    let data = prepare_preoptimize(problem, params)?;
    let model_build_seconds = build_start.elapsed().as_secs_f64();
    let mut state = build_state(problem, &data);
    let mut best = state.schedule.clone();
    let mut best_objective = state.objective;
    let mut rng = RandPcg64Mcg::new(RNG_SEED);
    let start_temperature = (state.objective / problem.blocks.len() as f64)
        .max(problem.weights.w1)
        .max(problem.weights.w3)
        .max(1.0);
    let end_temperature = start_temperature * END_TEMPERATURE_RATIO;
    let solve_start = Instant::now();
    let mut iter = 0usize;

    loop {
        let elapsed = solve_start.elapsed().as_secs_f64();
        if elapsed >= params.time_limit {
            break;
        }
        iter += 1;
        let progress = (elapsed / params.time_limit).clamp(0.0, 1.0);
        let temperature = start_temperature * (end_temperature / start_temperature).powf(progress);
        if rng.nextf() < SWAP_PROBABILITY {
            try_swap(problem, &data, &mut state, temperature, &mut rng);
        } else {
            try_relocate(problem, &data, &mut state, temperature, &mut rng);
        }
        if state.objective + 1e-9 < best_objective {
            best_objective = state.objective;
            best.clone_from(&state.schedule);
            eprintln!(
                "[{:.4}] annealing new best: iter={:8}, score={:.3}, z1={:.3}, z2={:.3}, z3={:.3}",
                solve_start.elapsed().as_secs_f64(),
                iter,
                state.objective,
                state.z1,
                state.z2,
                state.z3,
            );
        }
    }

    let solve_seconds = solve_start.elapsed().as_secs_f64();
    let (objective, z1, z2, z3) =
        evaluate_schedule(problem, &data.pref_penalty, &data.bay_load_scale, &best);
    Ok(PreoptimizeResult {
        status: PreoptimizeStatus::Annealing,
        objective,
        initial_objective: data.initial_objective,
        z1,
        z2,
        z3,
        blocks: best,
        horizon: data.horizon,
        variable_count: 0,
        constraint_count: 0,
        mip_gap: None,
        model_build_seconds,
        solve_seconds,
    })
}
