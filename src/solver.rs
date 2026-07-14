use crate::{
    insert::{InsertSearchParams, insert_greedy, try_place_block},
    precompute::Precompute,
    preoptimize::{PreoptimizeParams, PreoptimizePrecompute, preoptimize},
    solver_util::{
        NeighborKind, NeighborStats, format_neighbor_stats, sample_neighbor, schedule_to_solution,
        score_schedule, score13_block,
    },
    util::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
    *,
};
use rayon::prelude::*;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet, VecDeque},
    sync::Mutex,
    time::Instant,
};

macro_rules! log {
    ($timer:expr, $($arg:tt)*) => {
        eprintln!("[{:.4}] {}", $timer.elapsed_seconds(), format_args!($($arg)*))
    };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PreoptimizedBlock {
    pub bay_id: usize,
    pub entry_time: i64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PreoptimizeState {
    pub score: f64,
    pub blocks: Vec<PreoptimizedBlock>,
}

#[derive(Clone, Debug)]
pub struct OptimizeState {
    pub score: f64,
    pub blocks: Vec<ScheduledBlock>,
}

const RNG_SEED: u64 = 1;
const MAX_WORKER_COUNT: usize = 4;

const LOCAL_SEARCH_TIME_BUFFER_SECONDS: f64 = 3.;
const TEMP_WEIGHT_DIVISOR: f64 = 10.0;
const WORKER_TEMP_SCALE: f64 = 10.;
const BEST_EXCHANGE_INTERVAL: usize = 2_000;
const TABU_SIZE: usize = 4_096;

pub const PRECOMPUTE_ORIENTATION_NEIGHBOR_LIMIT: usize = 100;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_AREA_TOP_K: usize = 16;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_ALIGN_DELTA: i64 = 3;

const INITIAL_PREOPTIMIZE_ALPHA: f64 = 1.0;
const INITIAL_PREOPTIMIZE_BETA: f64 = 0.0;
const INITIAL_PREOPTIMIZE_TIME_RATIO: f64 = 0.1;
const INITIAL_PREOPTIMIZE_MAX_SECONDS: f64 = 10.0;
const GUIDED_PREOPTIMIZE_ALPHA: f64 = 0.5;
const GUIDED_PREOPTIMIZE_BETA: f64 = 0.0;
const GUIDED_PREOPTIMIZE_TIME_RATIO: f64 = 0.1;
const GUIDED_PREOPTIMIZE_MAX_SECONDS: f64 = 10.0;
const PREOPTIMIZE_DISTANCE_WEIGHT_SCALE: f64 = 1.0;
const PREOPTIMIZE_RNG_SEED: u64 = 2;
const PREOPTIMIZE_SWAP_PROBABILITY: f64 = 0.15;
const PREOPTIMIZE_BAD_BLOCK_SAMPLE_COUNT: usize = 8;
const PREOPTIMIZE_BAD_BLOCK_SELECT_PROBABILITY: f64 = 0.75;
const PREOPTIMIZE_MAX_RELOCATE_ATTEMPTS: usize = 8;
const PREOPTIMIZE_MAX_TIME_SHIFT: i64 = 10;
const PREOPTIMIZE_END_TEMPERATURE_RATIO: f64 = 1e-4;

const MIN_REMOVED_BLOCKS: usize = 7;
const MAX_REMOVED_BLOCKS: usize = 13;
const REMOVE_POOL_FACTOR: usize = 4;
const REMOVE_COUNT_SAMPLE_POWER: f64 = 2.0;
const REMOVE_SEED_PER_BLOCK: usize = 4;
const REMOVE_RANDOM_SEED_RATIO: f64 = 0.25;
const REMOVE_X_DISTANCE_WEIGHT_MAX: f64 = 3.0;
const REMOVE_Y_DISTANCE_WEIGHT_MAX: f64 = 3.0;
const INSERT_PARAMS: InsertSearchParams = InsertSearchParams { y_buffer: 10 };
const BAY_GREEDY_ORDER_TRIALS: usize = 32;
const BAY_GREEDY_MAX_DUPLICATE_TRIALS: usize = 128;

const SHIFT_MAX_SHIFT_X: i64 = 5;
const SHIFT_MAX_SHIFT_Y: i64 = 5;
const ROTATE_MAX_SHIFT_DELTA: i64 = 2;
const SWAP_NEIGHBOR_TOP_K: usize = 32;
const SWAP_MAX_SHIFT_DELTA: i64 = 2;
const MOVE_SAMPLE_BLOCKS: usize = 16;
const MOVE_SMALL_POOL_SIZE: usize = 8;

const NEIGHBOR_KIND_COUNT: usize = 5;
const NEIGHBOR_PROBS: &[(NeighborKind, f64)] = &[
    (NeighborKind::LargeReconstruct, 0.2),
    (NeighborKind::Shift, 8.),
    (NeighborKind::Move, 0.1),
    (NeighborKind::Rotate, 8.),
    (NeighborKind::Swap, 3.),
];

const RECONSTRUCT_WORKLOAD_WEIGHT_RANGE: (f64, f64) = (0., 1.);
const RECONSTRUCT_AREA_WEIGHT_RANGE: (f64, f64) = (-0.2, 1.);
const RECONSTRUCT_PREF_SPREAD_WEIGHT_RANGE: (f64, f64) = (0., 1.);
const RECONSTRUCT_DUE_URGENCY_WEIGHT_RANGE: (f64, f64) = (0., 1.);
const RECONSTRUCT_SLACK_URGENCY_WEIGHT_RANGE: (f64, f64) = (0., 1.);
const RECONSTRUCT_ORDER_RANDOM_WEIGHT_RANGE: (f64, f64) = (0., 0.5);

fn get_temp_scale(worker_id: usize, worker_count: usize) -> f64 {
    if worker_count <= 1 {
        return 1.0;
    }
    let ratio = worker_id as f64 / (worker_count - 1) as f64;
    1. + WORKER_TEMP_SCALE * ratio.powf(2.)
}

fn gen_rangef(rng: &mut impl Random, r: (f64, f64)) -> f64 {
    rng.gen_rangef(r.0, r.1)
}

fn sample_reconstruct_order_weights(rng: &mut impl Random) -> BlockOrderWeights {
    BlockOrderWeights {
        workload: gen_rangef(rng, RECONSTRUCT_WORKLOAD_WEIGHT_RANGE),
        area: gen_rangef(rng, RECONSTRUCT_AREA_WEIGHT_RANGE),
        pref_spread: gen_rangef(rng, RECONSTRUCT_PREF_SPREAD_WEIGHT_RANGE),
        due_urgency: gen_rangef(rng, RECONSTRUCT_DUE_URGENCY_WEIGHT_RANGE),
        slack_urgency: gen_rangef(rng, RECONSTRUCT_SLACK_URGENCY_WEIGHT_RANGE),
        random: gen_rangef(rng, RECONSTRUCT_ORDER_RANDOM_WEIGHT_RANGE),
    }
}

#[derive(Clone, Copy)]
struct BlockOrderWeights {
    workload: f64,
    area: f64,
    pref_spread: f64,
    due_urgency: f64,
    slack_urgency: f64,
    random: f64,
}

struct BlockOrderContext {
    max_workload: f64,
    max_area: f64,
    max_pref_spread: f64,
    max_due: i64,
    due_span: f64,
    max_slack: i64,
    slack_span: f64,
}

struct PrecedenceConstraints {
    befores: Vec<Vec<usize>>,
    afters: Vec<Vec<usize>>,
}

struct AnnealingResult {
    worker_id: usize,
    state: OptimizeState,
    current_score: f64,
    iter: usize,
    accepted: usize,
    improved: usize,
    neighbor_stats: [NeighborStats; NEIGHBOR_KIND_COUNT],
}

pub fn to_preoptimize_state(state: &OptimizeState) -> PreoptimizeState {
    let mut ordered = state.blocks.clone();
    ordered.sort_unstable_by_key(|scheduled| scheduled.block_id);
    PreoptimizeState {
        score: state.score,
        blocks: ordered
            .into_iter()
            .map(|scheduled| PreoptimizedBlock {
                bay_id: scheduled.bay_id,
                entry_time: scheduled.entry_time,
            })
            .collect(),
    }
}

fn make_preoptimize_params(
    alpha: f64,
    beta: f64,
    time_limit: f64,
    distance_weight: f64,
    rng_seed: u64,
) -> PreoptimizeParams {
    PreoptimizeParams {
        alpha,
        beta,
        time_limit,
        distance_weight,
        rng_seed,
        swap_probability: PREOPTIMIZE_SWAP_PROBABILITY,
        bad_block_sample_count: PREOPTIMIZE_BAD_BLOCK_SAMPLE_COUNT,
        bad_block_select_probability: PREOPTIMIZE_BAD_BLOCK_SELECT_PROBABILITY,
        max_relocate_attempts: PREOPTIMIZE_MAX_RELOCATE_ATTEMPTS,
        max_time_shift: PREOPTIMIZE_MAX_TIME_SHIFT,
        end_temperature_ratio: PREOPTIMIZE_END_TEMPERATURE_RATIO,
    }
}

fn preoptimize_time_limit(timelimit: f64, ratio: f64, max_seconds: f64) -> f64 {
    (timelimit * ratio).min(max_seconds).max(1e-4)
}

pub fn solve(problem: &Problem, timelimit: f64, timer: Timer) -> Result<Solution, String> {
    let deadline = timelimit - LOCAL_SEARCH_TIME_BUFFER_SECONDS;

    log!(timer, "building preoptimize precompute...");
    let preoptimize_pre = PreoptimizePrecompute::build(problem)?;
    log!(timer, "preoptimize precompute built");
    log!(timer, "building precompute...");
    let pre = Precompute::build(problem);
    log!(timer, "precompute built");

    let initial_time_limit = preoptimize_time_limit(
        timelimit,
        INITIAL_PREOPTIMIZE_TIME_RATIO,
        INITIAL_PREOPTIMIZE_MAX_SECONDS,
    )
    .min((deadline - timer.elapsed_seconds()).max(1e-4));
    let initial_abstract = preoptimize(
        problem,
        &preoptimize_pre,
        None,
        make_preoptimize_params(
            INITIAL_PREOPTIMIZE_ALPHA,
            INITIAL_PREOPTIMIZE_BETA,
            initial_time_limit,
            0.0,
            PREOPTIMIZE_RNG_SEED,
        ),
    )?;
    log!(
        timer,
        "initial abstract score: {:.3}",
        initial_abstract.score
    );
    let mut initial = build_optimize_state(problem, &pre, &initial_abstract)
        .ok_or_else(|| "failed to build initial optimize state".to_string())?;
    log!(timer, "initial optimize score: {:.3}", initial.score);

    let current_abstract = to_preoptimize_state(&initial);
    let guided_time_limit = preoptimize_time_limit(
        timelimit,
        GUIDED_PREOPTIMIZE_TIME_RATIO,
        GUIDED_PREOPTIMIZE_MAX_SECONDS,
    )
    .min(timelimit - timer.elapsed_seconds());
    let guided_abstract = preoptimize(
        problem,
        &preoptimize_pre,
        Some(&current_abstract),
        make_preoptimize_params(
            GUIDED_PREOPTIMIZE_ALPHA,
            GUIDED_PREOPTIMIZE_BETA,
            guided_time_limit,
            problem.weights.w3 * PREOPTIMIZE_DISTANCE_WEIGHT_SCALE,
            PREOPTIMIZE_RNG_SEED.wrapping_add(1),
        ),
    )?;
    log!(timer, "guided abstract score: {:.3}", guided_abstract.score);
    if guided_abstract.score + 1e-9 < current_abstract.score {
        if let Some(candidate) = build_optimize_state(problem, &pre, &guided_abstract) {
            log!(timer, "guided optimize score: {:.3}", candidate.score);
            if candidate.score + 1e-9 < initial.score {
                initial = candidate;
            }
        }
    }

    let best = run_global_annealing(problem, &pre, initial, deadline, timer)?;
    Ok(schedule_to_solution(&best.blocks))
}

fn run_global_annealing(
    problem: &Problem,
    pre: &Precompute,
    initial: OptimizeState,
    deadline: f64,
    timer: Timer,
) -> Result<OptimizeState, String> {
    let worker_count = rayon::current_num_threads().clamp(1, MAX_WORKER_COUNT);
    log!(timer, "annealing workers: {}", worker_count);
    let state = Mutex::new(initial.clone());
    let worker_timers = vec![timer; worker_count];
    let results: Vec<_> = worker_timers
        .into_par_iter()
        .enumerate()
        .map(|(worker_id, worker_timer)| {
            run_annealing_worker(
                problem,
                pre,
                &state,
                &initial,
                deadline,
                worker_id,
                worker_count,
                worker_timer,
            )
        })
        .collect::<Vec<AnnealingResult>>();

    let mut best: Option<OptimizeState> = None;
    let mut best_score = f64::INFINITY;
    for result in results {
        eprintln!(
            "[{:.4}] [id={}] iter={}, best={:.3}, accepted={}, improved={}, current={:.3}\nneighbor stats:\n{}",
            timer.elapsed_seconds(),
            result.worker_id,
            result.iter,
            result.state.score,
            result.accepted,
            result.improved,
            result.current_score,
            format_neighbor_stats(&result.neighbor_stats, NEIGHBOR_PROBS),
        );
        if result.state.score + 1e-9 < best_score {
            best_score = result.state.score;
            best = Some(result.state);
        }
    }
    best.ok_or_else(|| "no annealing worker result".to_string())
}

fn run_annealing_worker(
    problem: &Problem,
    pre: &Precompute,
    state: &Mutex<OptimizeState>,
    initial: &OptimizeState,
    deadline: f64,
    worker_id: usize,
    worker_count: usize,
    timer: Timer,
) -> AnnealingResult {
    let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(worker_id as u64));
    let temp_scale = get_temp_scale(worker_id, worker_count);
    let mut current = initial.blocks.clone();
    let mut current_score = initial.score;
    let mut best = current.clone();
    let mut best_score = current_score;
    let mut tabu_queue = VecDeque::with_capacity(TABU_SIZE);
    let mut tabu_set = HashSet::with_capacity(TABU_SIZE * 2);
    push_tabu(hash_schedule(&current), &mut tabu_queue, &mut tabu_set);
    let mut iter = 0usize;
    let mut accepted = 0usize;
    let mut improved = 0usize;
    let mut neighbor_stats = [NeighborStats::default(); NEIGHBOR_KIND_COUNT];
    let start = timer.elapsed_seconds();

    loop {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= deadline {
            break;
        }
        iter += 1;
        if iter % BEST_EXCHANGE_INTERVAL == 0 {
            let shared = state.lock().unwrap();
            if shared.score + problem.weights.w1 + 1e-9 < current_score {
                current = shared.blocks.clone();
                current_score = shared.score;
                push_tabu(hash_schedule(&current), &mut tabu_queue, &mut tabu_set);
                if shared.score + 1e-9 < best_score {
                    best = current.clone();
                    best_score = current_score;
                }
            }
        }

        let progress = (elapsed / (deadline - start).max(1e-4)).clamp(0.0, 1.0);
        let start_temp = (problem.weights.w1 / TEMP_WEIGHT_DIVISOR).max(1e-9);
        let end_temp = (problem.weights.w3 / TEMP_WEIGHT_DIVISOR).max(1e-9);
        let temp = temp_scale * start_temp * (end_temp / start_temp).powf(progress);

        let neighbor = sample_neighbor(&mut rng, NEIGHBOR_PROBS);
        let neighbor_idx = neighbor.index();
        let neighbor_start = Instant::now();
        neighbor_stats[neighbor_idx].selected += 1;
        let accept_threshold = current_score - temp * rng.nextf().ln();

        let candidate = match neighbor {
            NeighborKind::LargeReconstruct => {
                try_large_reconstruct(problem, pre, &current, &mut rng, accept_threshold)
            }
            NeighborKind::Shift => try_shift_neighbor(problem, pre, &current, &mut rng),
            NeighborKind::Move => try_move_neighbor(problem, pre, &current, &mut rng),
            NeighborKind::Rotate => try_rotate_neighbor(problem, pre, &current, &mut rng),
            NeighborKind::Swap => try_swap_neighbor(problem, pre, &current, &mut rng),
        };
        let Some(candidate) = candidate else {
            neighbor_stats[neighbor_idx].time_sec += neighbor_start.elapsed().as_secs_f64();
            continue;
        };
        neighbor_stats[neighbor_idx].succeeded += 1;

        let candidate_hash = hash_schedule(&candidate);
        let tabu = tabu_set.contains(&candidate_hash);
        let score = score_schedule(problem, pre, &candidate);
        if tabu && score + 1e-9 >= best_score {
            neighbor_stats[neighbor_idx].time_sec += neighbor_start.elapsed().as_secs_f64();
            continue;
        }

        let delta = score - current_score;
        let improved_current = delta < -1e-9;
        if improved_current {
            neighbor_stats[neighbor_idx].improved += 1;
            neighbor_stats[neighbor_idx].improved_delta_sum += -delta;
        }
        if score <= accept_threshold {
            current = candidate;
            current_score = score;
            push_tabu(candidate_hash, &mut tabu_queue, &mut tabu_set);
            accepted += 1;
            neighbor_stats[neighbor_idx].accepted += 1;

            if current_score + 1e-9 < best_score {
                log!(
                    timer,
                    "worker {} new best score: {:.3}",
                    worker_id,
                    current_score
                );
                best = current.clone();
                best_score = current_score;
                improved += 1;

                let mut shared = state.lock().unwrap();
                if current_score + 1e-9 < shared.score {
                    shared.score = current_score;
                    shared.blocks = current.clone();
                }
            }
        }
        neighbor_stats[neighbor_idx].time_sec += neighbor_start.elapsed().as_secs_f64();
    }

    AnnealingResult {
        worker_id,
        state: OptimizeState {
            score: best_score,
            blocks: best,
        },
        current_score,
        iter,
        accepted,
        improved,
        neighbor_stats,
    }
}

fn hash_order(order: &[usize]) -> u64 {
    let mut hash = 1469598103934665603u64;
    for &block_id in order {
        hash ^= block_id as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

fn mix_hash(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash = hash.wrapping_mul(1099511628211);
    hash
}

fn hash_scheduled_block(s: ScheduledBlock) -> u64 {
    let mut hash = 1469598103934665603u64;
    hash = mix_hash(hash, s.block_id as u64);
    hash = mix_hash(hash, s.bay_id as u64);
    hash = mix_hash(hash, s.orient_idx as u64);
    hash = mix_hash(hash, s.x as u64);
    hash = mix_hash(hash, s.y as u64);
    hash = mix_hash(hash, s.entry_time as u64);
    mix_hash(hash, s.exit_time as u64)
}

fn hash_schedule(schedule: &[ScheduledBlock]) -> u64 {
    let mut hash = mix_hash(1469598103934665603u64, schedule.len() as u64);
    for &s in schedule {
        hash ^= hash_scheduled_block(s);
    }
    hash
}

fn push_tabu(hash: u64, queue: &mut VecDeque<u64>, set: &mut HashSet<u64>) {
    if !set.insert(hash) {
        return;
    }
    queue.push_back(hash);
    if queue.len() > TABU_SIZE {
        if let Some(old_hash) = queue.pop_front() {
            set.remove(&old_hash);
        }
    }
}

fn try_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng).min(problem.blocks.len());

    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng);
    if removed_ids.is_empty() {
        return None;
    }
    let w = sample_reconstruct_order_weights(rng);
    sort_block_order(problem, pre, &mut removed_ids, w, rng);

    let mut removed_ordered = vec![None; removed_ids.len()];
    let mut base = Vec::with_capacity(schedule.len() - removed_ids.len());
    for &s in schedule {
        if let Some(pos) = removed_ids.iter().position(|&id| id == s.block_id) {
            removed_ordered[pos] = Some(s);
        } else {
            base.push(s);
        }
    }

    let mut cur = base;
    let mut loads = vec![0.0; problem.bays.len()];
    let mut fixed_score13 = 0.0;
    for &s in &cur {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
        fixed_score13 += score13_block(problem, pre, s);
    }

    for old in removed_ordered {
        if fixed_score13 > accept_threshold + 1e-9 {
            return None;
        }
        let old = old?;
        let scheduled = insert_greedy(
            problem,
            pre,
            old,
            i64::MIN,
            i64::MAX,
            &cur,
            &loads,
            INSERT_PARAMS,
            &pre.bay_order_by_pref[old.block_id],
        )?;
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        fixed_score13 += score13_block(problem, pre, scheduled);
        cur.push(scheduled);
    }

    Some(cur)
}

fn update_best(
    problem: &Problem,
    best: &mut Option<(i64, i64, ScheduledBlock)>,
    candidate: ScheduledBlock,
) {
    let block = &problem.blocks[candidate.block_id];
    let tardiness = (candidate.exit_time - block.due_date).max(0);
    if best
        .as_ref()
        .map_or(true, |&(best_tardiness, best_entry_time, _)| {
            (tardiness, candidate.entry_time) < (best_tardiness, best_entry_time)
        })
    {
        *best = Some((tardiness, candidate.entry_time, candidate));
    }
}

fn try_shift_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];

    let mut base = Vec::with_capacity(schedule.len());
    for (i, &s) in schedule.iter().enumerate() {
        if i != idx {
            base.push(s);
        }
    }

    let range = pre
        .collision
        .fit_range(old.bay_id, old.block_id, old.orient_idx)?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;

    for dist in (1..=SHIFT_MAX_SHIFT_X + SHIFT_MAX_SHIFT_Y).rev() {
        let min_abs_dx = (dist - SHIFT_MAX_SHIFT_Y).max(0);
        let max_abs_dx = dist.min(SHIFT_MAX_SHIFT_X);

        for abs_dx in min_abs_dx..=max_abs_dx {
            let abs_dy = dist - abs_dx;
            let x = old.x - abs_dx;
            let y = old.y - abs_dy;
            if !range.contains(x, y) {
                continue;
            }

            let Some(moved) = try_place_block(
                problem,
                pre,
                &base,
                old.block_id,
                old.bay_id,
                old.orient_idx,
                x,
                y,
                i64::MIN,
                i64::MAX,
            ) else {
                continue;
            };
            if moved == old {
                continue;
            }
            update_best(problem, &mut best, moved);
        }
    }

    let moved = best?.2;
    base.push(moved);
    Some(base)
}

fn try_rotate_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];
    let block = &problem.blocks[old.block_id];
    if block.shape.len() <= 1 {
        return None;
    }

    let mut base = Vec::with_capacity(schedule.len());
    for (i, &s) in schedule.iter().enumerate() {
        if i != idx {
            base.push(s);
        }
    }

    let mut best: Option<(i64, i64, ScheduledBlock)> = None;
    for &(orient_idx, dx, dy) in &pre.orientation_neighbors[old.block_id][old.orient_idx] {
        let Some(range) = pre
            .collision
            .fit_range(old.bay_id, old.block_id, orient_idx)
        else {
            continue;
        };

        for ddx in -ROTATE_MAX_SHIFT_DELTA..=ROTATE_MAX_SHIFT_DELTA {
            for ddy in -ROTATE_MAX_SHIFT_DELTA..=ROTATE_MAX_SHIFT_DELTA {
                let x = old.x + dx + ddx;
                let y = old.y + dy + ddy;
                if !range.contains(x, y) {
                    continue;
                }

                let Some(rotated) = try_place_block(
                    problem,
                    pre,
                    &base,
                    old.block_id,
                    old.bay_id,
                    orient_idx,
                    x,
                    y,
                    i64::MIN,
                    i64::MAX,
                ) else {
                    continue;
                };
                if rotated == old {
                    continue;
                }

                update_best(problem, &mut best, rotated);
            }
        }
    }

    let rotated = best?.2;
    base.push(rotated);
    Some(base)
}

fn try_swap_place(
    problem: &Problem,
    pre: &Precompute,
    old: ScheduledBlock,
    schedule: &[ScheduledBlock],
    bay_id: usize,
    orient_idx: usize,
    base_x: i64,
    base_y: i64,
) -> Option<ScheduledBlock> {
    let range = pre.collision.fit_range(bay_id, old.block_id, orient_idx)?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;

    for ddx in -SWAP_MAX_SHIFT_DELTA..=SWAP_MAX_SHIFT_DELTA {
        for ddy in -SWAP_MAX_SHIFT_DELTA..=SWAP_MAX_SHIFT_DELTA {
            let x = base_x + ddx;
            let y = base_y + ddy;
            if !range.contains(x, y) {
                continue;
            }

            let Some(scheduled) = try_place_block(
                problem,
                pre,
                schedule,
                old.block_id,
                bay_id,
                orient_idx,
                x,
                y,
                i64::MIN,
                i64::MAX,
            ) else {
                continue;
            };
            update_best(problem, &mut best, scheduled);
        }
    }

    Some(best?.2)
}

fn try_swap_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.len() < 2 {
        return None;
    }

    let a_idx = rng.gen_index(schedule.len());
    let a_old = schedule[a_idx];
    let candidates = &pre.other_block_neighbors[a_old.block_id][a_old.orient_idx];
    if candidates.is_empty() {
        return None;
    }
    let candidate_count = SWAP_NEIGHBOR_TOP_K.min(candidates.len());
    let candidate = candidates[rng.gen_index(candidate_count)];
    let b_idx = schedule
        .iter()
        .position(|s| s.block_id == candidate.block_id)?;
    if a_idx == b_idx {
        return None;
    }
    let b_old = schedule[b_idx];

    let mut cur = Vec::with_capacity(schedule.len());
    for (idx, &s) in schedule.iter().enumerate() {
        if idx != a_idx && idx != b_idx {
            cur.push(s);
        }
    }

    let a_base_x = b_old.x - candidate.dx;
    let a_base_y = b_old.y - candidate.dy;
    let a_new = try_swap_place(
        problem,
        pre,
        a_old,
        &cur,
        b_old.bay_id,
        a_old.orient_idx,
        a_base_x,
        a_base_y,
    )?;
    cur.push(a_new);

    let b_base_x = a_old.x + candidate.dx;
    let b_base_y = a_old.y + candidate.dy;
    let b_new = try_swap_place(
        problem,
        pre,
        b_old,
        &cur,
        a_old.bay_id,
        candidate.orient_idx,
        b_base_x,
        b_base_y,
    )?;
    cur.push(b_new);

    Some(cur)
}

fn try_move_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    fn move_obj13(problem: &Problem, pre: &Precompute, s: ScheduledBlock) -> f64 {
        let block = &problem.blocks[s.block_id];
        let tardiness = (s.exit_time - block.due_date).max(0) as f64;
        problem.weights.w1 * tardiness
            + problem.weights.w3 * pre.pref_penalty[s.block_id][s.bay_id] as f64
    }

    if schedule.is_empty() {
        return None;
    }

    let sample_count = MOVE_SAMPLE_BLOCKS.min(schedule.len());
    let mut indices: Vec<usize> = (0..schedule.len()).collect();
    rng.shuffle(&mut indices);
    indices.truncate(sample_count);
    indices.sort_by(|&a, &b| {
        pre.block_area[schedule[a].block_id]
            .total_cmp(&pre.block_area[schedule[b].block_id])
            .then(schedule[a].block_id.cmp(&schedule[b].block_id))
    });
    indices.truncate(MOVE_SMALL_POOL_SIZE.min(indices.len()));

    let idx = indices.into_iter().max_by(|&a, &b| {
        let sa = move_obj13(problem, pre, schedule[a]);
        let sb = move_obj13(problem, pre, schedule[b]);
        sa.total_cmp(&sb).then(
            pre.block_area[schedule[b].block_id].total_cmp(&pre.block_area[schedule[a].block_id]),
        )
    })?;
    let old = schedule[idx];

    let mut base = Vec::with_capacity(schedule.len());
    let mut loads = vec![0.0; problem.bays.len()];
    for (i, &s) in schedule.iter().enumerate() {
        if i == idx {
            continue;
        }
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
        base.push(s);
    }

    let scheduled = insert_greedy(
        problem,
        pre,
        old,
        i64::MIN,
        i64::MAX,
        &base,
        &loads,
        INSERT_PARAMS,
        &pre.bay_order_by_pref[old.block_id],
    )?;
    if scheduled == old {
        return None;
    }

    base.push(scheduled);
    Some(base)
}

fn sample_removed_count<R: Random>(rng: &mut R) -> usize {
    let span = MAX_REMOVED_BLOCKS - MIN_REMOVED_BLOCKS + 1;
    let u = rng.nextf().powf(REMOVE_COUNT_SAMPLE_POWER);
    MIN_REMOVED_BLOCKS + ((u * span as f64) as usize).min(span - 1)
}

fn choose_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    k: usize,
    rng: &mut R,
) -> Vec<usize> {
    fn scheduled_center(pre: &Precompute, s: ScheduledBlock) -> (f64, f64) {
        let (cx, cy) = pre.orientation_bbox_center[s.block_id][s.orient_idx];
        (s.x as f64 + cx, s.y as f64 + cy)
    }

    fn most_loaded_bay(pre: &Precompute, loads: &[f64]) -> Option<usize> {
        if loads.is_empty() {
            return None;
        }
        let mut best = 0;
        let mut best_load = f64::NEG_INFINITY;
        for (bay_id, &load) in loads.iter().enumerate() {
            let normalized = load * pre.bay_load_scale[bay_id];
            if normalized > best_load {
                best_load = normalized;
                best = bay_id;
            }
        }
        Some(best)
    }

    let n = schedule.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }

    let k = k.min(n);
    let mut ids: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();

    let mut loads = vec![0.0; problem.bays.len()];
    for s in schedule {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }
    let heavy_bay = most_loaded_bay(pre, &loads);

    let remove_x_distance_weight = rng.gen_rangef(0.0, REMOVE_X_DISTANCE_WEIGHT_MAX);
    let remove_y_distance_weight = rng.gen_rangef(0.0, REMOVE_Y_DISTANCE_WEIGHT_MAX);

    let mut badness = vec![0i64; problem.blocks.len()];
    for s in schedule {
        let block = &problem.blocks[s.block_id];
        let tardiness = (s.exit_time - block.due_date).max(0);
        let pref_penalty = pre.pref_penalty[s.block_id][s.bay_id];
        let score = tardiness as f64 * problem.weights.w1
            + pref_penalty as f64 * problem.weights.w3
            + if Some(s.bay_id) == heavy_bay {
                problem.weights.w2
            } else {
                0.
            };
        badness[s.block_id] = score as i64;
    }

    let mut by_block = vec![None; problem.blocks.len()];
    for &s in schedule {
        by_block[s.block_id] = Some(s);
    }

    fn push_selected(selected: &mut Vec<usize>, used: &mut [bool], block_id: usize, k: usize) {
        if selected.len() < k && !used[block_id] {
            selected.push(block_id);
            used[block_id] = true;
        }
    }

    rng.shuffle(&mut ids);
    ids.sort_by_key(|&block_id| Reverse(badness[block_id]));
    let pool_len = (k * REMOVE_POOL_FACTOR).min(ids.len()).max(k);
    ids.truncate(pool_len);
    let bad_pool = ids;

    let mut selected = Vec::with_capacity(k);
    let mut used = vec![false; problem.blocks.len()];
    let seed_count = k.div_ceil(REMOVE_SEED_PER_BLOCK);
    let mut random_seed_count = 0;
    for _ in 0..seed_count {
        if rng.nextf() < REMOVE_RANDOM_SEED_RATIO {
            random_seed_count += 1;
        }
    }
    let bad_seed_count = seed_count - random_seed_count;

    let mut seed_pool = bad_pool.clone();
    rng.shuffle(&mut seed_pool);
    for &block_id in seed_pool.iter().take(bad_seed_count) {
        push_selected(&mut selected, &mut used, block_id, k);
    }

    let mut random_seed_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut random_seed_pool);
    let random_seed_end = selected.len() + random_seed_count;
    for block_id in random_seed_pool {
        if selected.len() >= random_seed_end {
            break;
        }
        push_selected(&mut selected, &mut used, block_id, k);
    }

    let seeds = selected.clone();
    for (seed_index, &seed_id) in seeds.iter().enumerate() {
        if selected.len() >= k {
            break;
        }
        let Some(seed) = by_block[seed_id] else {
            continue;
        };
        let seeds_left = seeds.len() - seed_index;
        let need = (k - selected.len() + seeds_left - 1) / seeds_left;
        let (sx, sy) = scheduled_center(pre, seed);
        let st = seed.entry_time as f64;
        let mut neighbors: Vec<(f64, usize)> = schedule
            .iter()
            .filter(|s| s.bay_id == seed.bay_id && !used[s.block_id])
            .map(|&s| {
                let (x, y) = scheduled_center(pre, s);
                let dx = x - sx;
                let dy = y - sy;
                let dt = s.entry_time as f64 - st;
                (
                    remove_x_distance_weight * dx * dx
                        + remove_y_distance_weight * dy * dy
                        + dt * dt,
                    s.block_id,
                )
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, block_id) in neighbors.into_iter().take(need) {
            push_selected(&mut selected, &mut used, block_id, k);
        }
    }

    let mut fill_pool = bad_pool;
    rng.shuffle(&mut fill_pool);
    for block_id in fill_pool {
        push_selected(&mut selected, &mut used, block_id, k);
    }

    selected
}

fn block_pref_spread(problem: &Problem, block_id: usize) -> i64 {
    let prefs = &problem.blocks[block_id].bay_preferences;
    let min_pref = prefs.iter().copied().min().unwrap_or(0);
    let max_pref = prefs.iter().copied().max().unwrap_or(min_pref);
    max_pref - min_pref
}

fn block_slack(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    block.due_date - block.release_time - block.processing_time
}

fn build_block_order_context(
    problem: &Problem,
    pre: &Precompute,
    order: &[usize],
) -> BlockOrderContext {
    let max_workload = order
        .iter()
        .map(|&block_id| problem.blocks[block_id].workload as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let max_area = order
        .iter()
        .map(|&block_id| pre.block_area[block_id])
        .fold(0.0, f64::max)
        .max(1.0);
    let max_pref_spread = order
        .iter()
        .map(|&block_id| block_pref_spread(problem, block_id) as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let min_due = order
        .iter()
        .map(|&block_id| problem.blocks[block_id].due_date)
        .min()
        .unwrap_or(0);
    let max_due = order
        .iter()
        .map(|&block_id| problem.blocks[block_id].due_date)
        .max()
        .unwrap_or(min_due);
    let min_slack = order
        .iter()
        .map(|&block_id| block_slack(problem, block_id))
        .min()
        .unwrap_or(0);
    let max_slack = order
        .iter()
        .map(|&block_id| block_slack(problem, block_id))
        .max()
        .unwrap_or(min_slack);

    BlockOrderContext {
        max_workload,
        max_area,
        max_pref_spread,
        max_due,
        due_span: (max_due - min_due).max(1) as f64,
        max_slack,
        slack_span: (max_slack - min_slack).max(1) as f64,
    }
}

fn block_order_score(
    problem: &Problem,
    pre: &Precompute,
    ctx: &BlockOrderContext,
    weights: BlockOrderWeights,
    block_id: usize,
) -> f64 {
    let block = &problem.blocks[block_id];
    let workload_norm = block.workload as f64 / ctx.max_workload;
    let area_norm = pre.block_area[block_id] / ctx.max_area;
    let pref_spread_norm = block_pref_spread(problem, block_id) as f64 / ctx.max_pref_spread;
    let due_urgency = (ctx.max_due - block.due_date) as f64 / ctx.due_span;
    let slack_urgency = (ctx.max_slack - block_slack(problem, block_id)) as f64 / ctx.slack_span;

    weights.workload * workload_norm
        + weights.area * area_norm
        + weights.pref_spread * pref_spread_norm
        + weights.due_urgency * due_urgency
        + weights.slack_urgency * slack_urgency
}

fn sort_block_order<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    order: &mut [usize],
    weights: BlockOrderWeights,
    rng: &mut R,
) {
    let ctx = build_block_order_context(problem, pre, order);
    let mut random_scores = vec![0.0; problem.blocks.len()];
    for &block_id in order.iter() {
        random_scores[block_id] = rng.gen_rangef(0., weights.random);
    }
    order.sort_by(|&a, &b| {
        let score_a = block_order_score(problem, pre, &ctx, weights, a) + random_scores[a];
        let score_b = block_order_score(problem, pre, &ctx, weights, b) + random_scores[b];
        score_b
            .total_cmp(&score_a)
            .then(block_slack(problem, a).cmp(&block_slack(problem, b)))
            .then(problem.blocks[a].due_date.cmp(&problem.blocks[b].due_date))
            .then(pre.block_area[b].total_cmp(&pre.block_area[a]))
            .then(problem.blocks[b].workload.cmp(&problem.blocks[a].workload))
            .then(block_pref_spread(problem, b).cmp(&block_pref_spread(problem, a)))
            .then(a.cmp(&b))
    });
}

fn build_precedence_constraints(
    problem: &Problem,
    state: &PreoptimizeState,
) -> PrecedenceConstraints {
    let mut befores = vec![Vec::new(); problem.blocks.len()];
    let mut afters = vec![Vec::new(); problem.blocks.len()];
    for i in 0..problem.blocks.len() {
        for j in i + 1..problem.blocks.len() {
            let si = state.blocks[i];
            let sj = state.blocks[j];
            if si.bay_id != sj.bay_id {
                continue;
            }
            let end_i = si.entry_time + problem.blocks[i].processing_time;
            let end_j = sj.entry_time + problem.blocks[j].processing_time;
            let edge = if end_i <= sj.entry_time {
                Some((i, j))
            } else if end_j <= si.entry_time {
                Some((j, i))
            } else {
                None
            };
            if let Some((before, after)) = edge {
                afters[before].push(after);
                befores[after].push(before);
            }
        }
    }
    PrecedenceConstraints { befores, afters }
}

fn build_topological_order<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    bay_block_ids: &[usize],
    constraints: &PrecedenceConstraints,
    weights: BlockOrderWeights,
    rng: &mut R,
) -> Vec<usize> {
    let mut priority_order = bay_block_ids.to_vec();
    sort_block_order(problem, pre, &mut priority_order, weights, rng);
    let mut priority_rank = vec![usize::MAX; problem.blocks.len()];
    for (rank, &block_id) in priority_order.iter().enumerate() {
        priority_rank[block_id] = rank;
    }

    let mut indegree = vec![0usize; problem.blocks.len()];
    let mut ready = BinaryHeap::new();
    for &block_id in bay_block_ids {
        indegree[block_id] = constraints.befores[block_id].len();
        if indegree[block_id] == 0 {
            ready.push(Reverse((priority_rank[block_id], block_id)));
        }
    }

    let mut order = Vec::with_capacity(bay_block_ids.len());
    while let Some(Reverse((_, block_id))) = ready.pop() {
        order.push(block_id);
        for &after in &constraints.afters[block_id] {
            indegree[after] -= 1;
            if indegree[after] == 0 {
                ready.push(Reverse((priority_rank[after], after)));
            }
        }
    }
    debug_assert_eq!(order.len(), bay_block_ids.len());
    order
}

fn build_bay_schedule(
    problem: &Problem,
    pre: &Precompute,
    bay_id: usize,
    order: &[usize],
    constraints: &PrecedenceConstraints,
) -> Option<Vec<ScheduledBlock>> {
    let mut schedule = Vec::with_capacity(order.len());
    let mut scheduled_by_id: Vec<Option<ScheduledBlock>> = vec![None; problem.blocks.len()];
    let mut loads = vec![0.0; problem.bays.len()];
    let bay_order = [bay_id];

    for &block_id in order {
        let block = &problem.blocks[block_id];
        let min_entry_time = constraints.befores[block_id]
            .iter()
            .map(|&before| scheduled_by_id[before].unwrap().exit_time)
            .fold(block.release_time, i64::max);
        let original = ScheduledBlock {
            block_id,
            bay_id,
            orient_idx: 0,
            x: 0,
            y: 0,
            entry_time: block.release_time,
            exit_time: block.release_time + block.processing_time,
        };
        let scheduled = insert_greedy(
            problem,
            pre,
            original,
            min_entry_time,
            i64::MAX,
            &schedule,
            &loads,
            INSERT_PARAMS,
            &bay_order,
        )?;
        loads[bay_id] += block.workload as f64;
        scheduled_by_id[block_id] = Some(scheduled);
        schedule.push(scheduled);
    }
    Some(schedule)
}

fn bay_tardiness(problem: &Problem, schedule: &[ScheduledBlock]) -> i64 {
    schedule
        .iter()
        .map(|scheduled| (scheduled.exit_time - problem.blocks[scheduled.block_id].due_date).max(0))
        .sum()
}

pub fn build_optimize_state(
    problem: &Problem,
    pre: &Precompute,
    state: &PreoptimizeState,
) -> Option<OptimizeState> {
    let constraints = build_precedence_constraints(problem, state);
    let mut blocks_by_bay = vec![Vec::new(); problem.bays.len()];
    for (block_id, block) in state.blocks.iter().enumerate() {
        blocks_by_bay[block.bay_id].push(block_id);
    }

    let mut blocks = Vec::with_capacity(problem.blocks.len());
    for (bay_id, bay_block_ids) in blocks_by_bay.iter().enumerate() {
        if bay_block_ids.is_empty() {
            continue;
        }
        let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(20_000).wrapping_add(bay_id as u64));
        let mut seen_order_hashes = HashSet::new();
        let mut duplicate_trials = 0;
        let mut best: Option<(i64, Vec<ScheduledBlock>)> = None;
        while seen_order_hashes.len() < BAY_GREEDY_ORDER_TRIALS
            && duplicate_trials < BAY_GREEDY_MAX_DUPLICATE_TRIALS
        {
            let weights = sample_reconstruct_order_weights(&mut rng);
            let order = build_topological_order(
                problem,
                pre,
                bay_block_ids,
                &constraints,
                weights,
                &mut rng,
            );
            if !seen_order_hashes.insert(hash_order(&order)) {
                duplicate_trials += 1;
                continue;
            }
            let Some(schedule) = build_bay_schedule(problem, pre, bay_id, &order, &constraints)
            else {
                continue;
            };
            let tardiness = bay_tardiness(problem, &schedule);
            if best
                .as_ref()
                .is_none_or(|(best_tardiness, _)| tardiness < *best_tardiness)
            {
                best = Some((tardiness, schedule));
            }
        }
        blocks.extend(best?.1);
    }

    let score = score_schedule(problem, pre, &blocks);
    Some(OptimizeState { score, blocks })
}
