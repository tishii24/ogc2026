use crate::{
    insert::{InsertMode, InsertSearchParams, insert_greedy, try_place_block},
    precompute::Precompute,
    preoptimize::{PreoptimizeParams, PreoptimizedBlock},
    preoptimize2::preoptimize_annealing,
    solver_util::{
        NeighborKind, NeighborStats, format_neighbor_stats, guidance_block_distance,
        guidance_distance, sample_neighbor, schedule_to_solution, score_schedule, score13_block,
    },
    util::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
    *,
};
use rayon::prelude::*;
use std::{
    collections::{HashSet, VecDeque},
    sync::Mutex,
    time::Instant,
};

macro_rules! log {
    ($timer:expr, $($arg:tt)*) => {
        eprintln!("[{:.4}] {}", $timer.elapsed_seconds(), format_args!($($arg)*))
    };
}

const RNG_SEED: u64 = 1;
const MAX_WORKER_COUNT: usize = 4;

const LOCAL_SEARCH_TIME_BUFFER_SECONDS: f64 = 3.;
const TEMP_WEIGHT_DIVISOR: f64 = 10.0;
const WORKER_TEMP_SCALE: f64 = 10.;
const BEST_EXCHANGE_INTERVAL: usize = 2_000;
const TABU_SIZE: usize = 4_096;

const PREOPTIMIZE_TIME_RATIO: f64 = 0.1;
const MAX_PREOPTIMIZE_SECONDS: f64 = 30.;
const PREOPTIMIZE_ALPHA: f64 = 0.8;
const PREOPTIMIZE_BETA: f64 = 1.;
const GUIDED_PHASE_RATIO: f64 = 0.5;
const GUIDE_BAY_MISMATCH_PENALTY: f64 = 1_000_000.;
const GUIDE_END_TEMP_RATIO: f64 = 1e-3;

pub const PRECOMPUTE_ORIENTATION_NEIGHBOR_LIMIT: usize = 100;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_AREA_TOP_K: usize = 16;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_ALIGN_DELTA: i64 = 3;

const MAX_INITIAL_SEARCH_SECONDS: f64 = 30.;
const BASE_INITIAL_SEARCH_TIME_RATIO: f64 = 0.3;
const INITIAL_GOOD_WEIGHT_POOL_SIZE: usize = 32;
const INITIAL_EXPLOIT_PROB: f64 = 0.25;
const INITIAL_WEIGHT_MUTATION_SCALE: f64 = 0.2;

const MIN_REMOVED_BLOCKS: usize = 7;
const MAX_REMOVED_BLOCKS: usize = 13;
const REMOVE_POOL_FACTOR: usize = 4;
const REMOVE_COUNT_SAMPLE_POWER: f64 = 2.0;
const REMOVE_SEED_PER_BLOCK: usize = 4;
const REMOVE_RANDOM_SEED_RATIO: f64 = 0.25;
const REMOVE_X_DISTANCE_WEIGHT_MAX: f64 = 3.0;
const REMOVE_Y_DISTANCE_WEIGHT_MAX: f64 = 3.0;
const INSERT_PARAMS: InsertSearchParams = InsertSearchParams { y_buffer: 10 };

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

const INITIAL_WORKLOAD_WEIGHT_RANGE: (f64, f64) = (0., 0.1);
const INITIAL_AREA_WEIGHT_RANGE: (f64, f64) = (-0.2, 0.2);
const INITIAL_PREF_SPREAD_WEIGHT_RANGE: (f64, f64) = (0., 0.1);
const INITIAL_DUE_URGENCY_WEIGHT_RANGE: (f64, f64) = (0., 1.);
const INITIAL_SLACK_URGENCY_WEIGHT_RANGE: (f64, f64) = (0., 1.);
const INITIAL_ORDER_RANDOM_WEIGHT_RANGE: (f64, f64) = (0., 0.2);

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

fn sample_initial_order_weights(rng: &mut impl Random) -> BlockOrderWeights {
    BlockOrderWeights {
        workload: gen_rangef(rng, INITIAL_WORKLOAD_WEIGHT_RANGE),
        area: gen_rangef(rng, INITIAL_AREA_WEIGHT_RANGE),
        pref_spread: gen_rangef(rng, INITIAL_PREF_SPREAD_WEIGHT_RANGE),
        due_urgency: gen_rangef(rng, INITIAL_DUE_URGENCY_WEIGHT_RANGE),
        slack_urgency: gen_rangef(rng, INITIAL_SLACK_URGENCY_WEIGHT_RANGE),
        random: gen_rangef(rng, INITIAL_ORDER_RANDOM_WEIGHT_RANGE),
    }
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

fn mutate_initial_order_weights(
    weights: BlockOrderWeights,
    rng: &mut impl Random,
) -> BlockOrderWeights {
    fn mutate(value: f64, r: (f64, f64), rng: &mut impl Random) -> f64 {
        let width = (r.1 - r.0) * INITIAL_WEIGHT_MUTATION_SCALE;
        (value + rng.gen_rangef(-width, width)).clamp(r.0, r.1)
    }

    BlockOrderWeights {
        workload: mutate(weights.workload, INITIAL_WORKLOAD_WEIGHT_RANGE, rng),
        area: mutate(weights.area, INITIAL_AREA_WEIGHT_RANGE, rng),
        pref_spread: mutate(weights.pref_spread, INITIAL_PREF_SPREAD_WEIGHT_RANGE, rng),
        due_urgency: mutate(weights.due_urgency, INITIAL_DUE_URGENCY_WEIGHT_RANGE, rng),
        slack_urgency: mutate(
            weights.slack_urgency,
            INITIAL_SLACK_URGENCY_WEIGHT_RANGE,
            rng,
        ),
        random: mutate(weights.random, INITIAL_ORDER_RANDOM_WEIGHT_RANGE, rng),
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

struct State {
    score: f64,
    schedule: Vec<ScheduledBlock>,
}

struct InitialSearchState {
    best: Option<(f64, f64, Vec<ScheduledBlock>)>,
    good_weights: Vec<(f64, f64, BlockOrderWeights)>,
    seen_order_hashes: HashSet<u64>,
}

struct AnnealingResult {
    worker_id: usize,
    schedule: Vec<ScheduledBlock>,
    score: f64,
    current_score: f64,
    iter: usize,
    accepted: usize,
    improved: usize,
    neighbor_stats: [NeighborStats; NEIGHBOR_KIND_COUNT],
}

fn build_guidance(
    problem: &Problem,
    timelimit: f64,
    timer: Timer,
) -> Result<Vec<PreoptimizedBlock>, String> {
    let time_limit = (PREOPTIMIZE_TIME_RATIO * timelimit).min(MAX_PREOPTIMIZE_SECONDS);
    let result = preoptimize_annealing(
        problem,
        PreoptimizeParams {
            alpha: PREOPTIMIZE_ALPHA,
            beta: PREOPTIMIZE_BETA,
            time_limit,
            horizon_margin: 0,
        },
    )?;
    if result.blocks.len() != problem.blocks.len() {
        return Err(format!(
            "guidance block count mismatch: got {}, expected {}",
            result.blocks.len(),
            problem.blocks.len()
        ));
    }
    for (block_id, target) in result.blocks.iter().enumerate() {
        if target.bay_id >= problem.bays.len() {
            return Err(format!(
                "guidance block {block_id} has invalid bay {}",
                target.bay_id
            ));
        }
        if target.entry_time < problem.blocks[block_id].release_time {
            return Err(format!(
                "guidance block {block_id} starts before its release time"
            ));
        }
    }
    log!(
        timer,
        "guidance: score={:.3}, initial={:.3}, z1={:.3}, z2={:.3}, z3={:.3}",
        result.objective,
        result.initial_objective,
        result.z1,
        result.z2,
        result.z3
    );
    Ok(result.blocks)
}

pub fn solve(problem: &Problem, timelimit: f64, timer: Timer) -> Result<Solution, String> {
    log!(timer, "building guidance...");
    let guidance = build_guidance(problem, timelimit, timer)?;
    log!(timer, "guidance built");

    log!(timer, "building precompute...");
    let pre = Precompute::build(problem);
    log!(timer, "precompute built");

    let deadline = timelimit - LOCAL_SEARCH_TIME_BUFFER_SECONDS;
    let worker_count = rayon::current_num_threads().clamp(1, MAX_WORKER_COUNT);
    log!(timer, "annealing workers: {}", worker_count);

    let (initial, initial_score, initial_distance) =
        search_initial_schedule(problem, &pre, &guidance, timelimit, timer, worker_count)?;
    log!(
        timer,
        "best initial: distance={:.3}, score={:.3}",
        initial_distance,
        initial_score
    );

    let state = Mutex::new(State {
        score: initial_score,
        schedule: initial.clone(),
    });
    let worker_timers = vec![timer; worker_count];
    let results: Vec<_> = worker_timers
        .into_par_iter()
        .enumerate()
        .map(|(worker_id, worker_timer)| {
            run_annealing_worker(
                problem,
                &pre,
                &state,
                &initial,
                &guidance,
                deadline,
                worker_id,
                worker_count,
                worker_timer,
            )
        })
        .collect::<Vec<AnnealingResult>>();

    let mut best: Option<Vec<ScheduledBlock>> = None;
    let mut best_score = f64::INFINITY;
    for result in results {
        eprintln!(
            "[{:.4}] [id={}] iter={}, best={:.3}, accepted={}, improved={}, current={:.3}\nneighbor stats:\n{}",
            timer.elapsed_seconds(),
            result.worker_id,
            result.iter,
            result.score,
            result.accepted,
            result.improved,
            result.current_score,
            format_neighbor_stats(&result.neighbor_stats, NEIGHBOR_PROBS),
        );
        if result.score + 1e-9 < best_score {
            best_score = result.score;
            best = Some(result.schedule);
        }
    }

    let best = best.ok_or_else(|| "no annealing worker result".to_string())?;
    Ok(schedule_to_solution(&best))
}

fn run_annealing_worker(
    problem: &Problem,
    pre: &Precompute,
    state: &Mutex<State>,
    initial: &[ScheduledBlock],
    guidance: &[PreoptimizedBlock],
    deadline: f64,
    worker_id: usize,
    worker_count: usize,
    timer: Timer,
) -> AnnealingResult {
    let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(worker_id as u64));
    let temp_scale = get_temp_scale(worker_id, worker_count);
    let mut current = initial.to_vec();
    let mut current_eval = guidance_distance(&current, guidance, GUIDE_BAY_MISMATCH_PENALTY);
    let mut best_guided_eval = current_eval;
    let mut best = current.clone();
    let mut best_score = score_schedule(problem, pre, &current);
    let mut tabu_queue = VecDeque::with_capacity(TABU_SIZE);
    let mut tabu_set = HashSet::with_capacity(TABU_SIZE * 2);
    push_tabu(hash_schedule(&current), &mut tabu_queue, &mut tabu_set);
    let mut iter = 0usize;
    let mut accepted = 0usize;
    let mut improved = 0usize;
    let mut neighbor_stats = [NeighborStats::default(); NEIGHBOR_KIND_COUNT];
    let start = timer.elapsed_seconds();
    let guided_deadline = start + GUIDED_PHASE_RATIO * (deadline - start).max(0.0);
    let guide_start_temp = (current_eval / problem.blocks.len().max(1) as f64).max(1.0);
    let guide_end_temp = guide_start_temp * GUIDE_END_TEMP_RATIO;
    let mut guided_phase = true;

    loop {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= deadline {
            break;
        }
        iter += 1;
        if guided_phase && elapsed >= guided_deadline {
            guided_phase = false;
            current_eval = score_schedule(problem, pre, &current);
            tabu_queue.clear();
            tabu_set.clear();
            push_tabu(hash_schedule(&current), &mut tabu_queue, &mut tabu_set);
            log!(
                timer,
                "worker {} switch to actual score: distance={:.3}, score={:.3}",
                worker_id,
                best_guided_eval,
                current_eval
            );
        }
        if !guided_phase && iter % BEST_EXCHANGE_INTERVAL == 0 {
            let shared = state.lock().unwrap();
            if shared.score + problem.weights.w1 + 1e-9 < current_eval {
                current = shared.schedule.clone();
                current_eval = shared.score;
                push_tabu(hash_schedule(&current), &mut tabu_queue, &mut tabu_set);
                if shared.score + 1e-9 < best_score {
                    best = current.clone();
                    best_score = shared.score;
                }
            }
        }

        let temp = if guided_phase {
            let progress =
                ((elapsed - start) / (guided_deadline - start).max(1e-4)).clamp(0.0, 1.0);
            temp_scale * guide_start_temp * (guide_end_temp / guide_start_temp).powf(progress)
        } else {
            let progress = ((elapsed - guided_deadline) / (deadline - guided_deadline).max(1e-4))
                .clamp(0.0, 1.0);
            let start_temp = (problem.weights.w1 / TEMP_WEIGHT_DIVISOR).max(1e-9);
            let end_temp = (problem.weights.w3 / TEMP_WEIGHT_DIVISOR).max(1e-9);
            temp_scale * start_temp * (end_temp / start_temp).powf(progress)
        };
        let active_guidance = guided_phase.then_some(guidance);

        let neighbor = sample_neighbor(&mut rng, NEIGHBOR_PROBS);
        let neighbor_idx = neighbor.index();
        let neighbor_start = Instant::now();
        neighbor_stats[neighbor_idx].selected += 1;
        let accept_threshold = current_eval - temp * rng.nextf().ln();

        let candidate = match neighbor {
            NeighborKind::LargeReconstruct => try_large_reconstruct(
                problem,
                pre,
                &current,
                &mut rng,
                (!guided_phase).then_some(accept_threshold),
                active_guidance,
            ),
            NeighborKind::Shift => {
                try_shift_neighbor(problem, pre, &current, &mut rng, active_guidance)
            }
            NeighborKind::Move => {
                try_move_neighbor(problem, pre, &current, &mut rng, active_guidance)
            }
            NeighborKind::Rotate => {
                try_rotate_neighbor(problem, pre, &current, &mut rng, active_guidance)
            }
            NeighborKind::Swap => {
                try_swap_neighbor(problem, pre, &current, &mut rng, active_guidance)
            }
        };
        let Some(candidate) = candidate else {
            neighbor_stats[neighbor_idx].time_sec += neighbor_start.elapsed().as_secs_f64();
            continue;
        };
        neighbor_stats[neighbor_idx].succeeded += 1;

        let candidate_hash = hash_schedule(&candidate);
        let tabu = tabu_set.contains(&candidate_hash);
        let candidate_eval = if guided_phase {
            guidance_distance(&candidate, guidance, GUIDE_BAY_MISMATCH_PENALTY)
        } else {
            score_schedule(problem, pre, &candidate)
        };
        let candidate_actual_score = if guided_phase {
            score_schedule(problem, pre, &candidate)
        } else {
            candidate_eval
        };
        let aspiration = if guided_phase {
            candidate_eval + 1e-9 < best_guided_eval
        } else {
            candidate_actual_score + 1e-9 < best_score
        };
        if tabu && !aspiration {
            neighbor_stats[neighbor_idx].time_sec += neighbor_start.elapsed().as_secs_f64();
            continue;
        }

        let delta = candidate_eval - current_eval;
        if delta < -1e-9 {
            neighbor_stats[neighbor_idx].improved += 1;
            neighbor_stats[neighbor_idx].improved_delta_sum += -delta;
        }
        if candidate_eval <= accept_threshold {
            current = candidate;
            current_eval = candidate_eval;
            push_tabu(candidate_hash, &mut tabu_queue, &mut tabu_set);
            accepted += 1;
            neighbor_stats[neighbor_idx].accepted += 1;

            if guided_phase && current_eval + 1e-9 < best_guided_eval {
                best_guided_eval = current_eval;
                log!(
                    timer,
                    "worker {} new best distance: {:.3}",
                    worker_id,
                    current_eval
                );
            }
            if candidate_actual_score + 1e-9 < best_score {
                log!(
                    timer,
                    "worker {} new best score: {:.3}",
                    worker_id,
                    candidate_actual_score
                );
                best = current.clone();
                best_score = candidate_actual_score;
                improved += 1;

                let mut shared = state.lock().unwrap();
                if candidate_actual_score + 1e-9 < shared.score {
                    shared.score = candidate_actual_score;
                    shared.schedule = current.clone();
                }
            }
        }
        neighbor_stats[neighbor_idx].time_sec += neighbor_start.elapsed().as_secs_f64();
    }

    AnnealingResult {
        worker_id,
        schedule: best,
        score: best_score,
        current_score: score_schedule(problem, pre, &current),
        iter,
        accepted,
        improved,
        neighbor_stats,
    }
}

fn search_initial_schedule(
    problem: &Problem,
    pre: &Precompute,
    guidance: &[PreoptimizedBlock],
    timelimit: f64,
    timer: Timer,
    worker_count: usize,
) -> Result<(Vec<ScheduledBlock>, f64, f64), String> {
    let search_seconds = MAX_INITIAL_SEARCH_SECONDS.min(BASE_INITIAL_SEARCH_TIME_RATIO * timelimit);
    let search_deadline = timer.elapsed_seconds() + search_seconds;
    log!(timer, "initial search seconds: {:.3}", search_seconds);

    let initial_state = Mutex::new(InitialSearchState {
        best: None,
        good_weights: Vec::with_capacity(INITIAL_GOOD_WEIGHT_POOL_SIZE),
        seen_order_hashes: HashSet::new(),
    });
    (0..worker_count).into_par_iter().for_each(|worker_id| {
        let mut trial = 0;
        let mut rng =
            RandPcg64Mcg::new(RNG_SEED.wrapping_add(10_000).wrapping_add(worker_id as u64));

        loop {
            if timer.elapsed_seconds() >= search_deadline
                && initial_state.lock().unwrap().best.is_some()
            {
                break;
            }

            let weights = sample_initial_search_weights(&mut rng, &initial_state);
            let order = build_initial_order(problem, pre, weights, &mut rng);
            let order_hash = hash_order(&order);
            {
                let mut state = initial_state.lock().unwrap();
                if !state.seen_order_hashes.insert(order_hash) {
                    continue;
                }
            }

            let Some(schedule) = build_initial_schedule_with_order(problem, pre, &order, guidance)
            else {
                continue;
            };
            let distance = guidance_distance(&schedule, guidance, GUIDE_BAY_MISMATCH_PENALTY);
            let score = score_schedule(problem, pre, &schedule);
            let mut state = initial_state.lock().unwrap();
            if state
                .best
                .as_ref()
                .is_none_or(|(best_distance, best_score, _)| {
                    distance
                        .total_cmp(best_distance)
                        .then(score.total_cmp(best_score))
                        .is_lt()
                })
            {
                log!(
                    timer,
                    "update initial: worker {} distance={:.3}, score={:.3}",
                    worker_id,
                    distance,
                    score
                );
                state.best = Some((distance, score, schedule));
            }
            update_good_weight_pool(&mut state.good_weights, distance, score, weights);

            trial += 1;
        }

        log!(timer, "worker {} trial {:4}", worker_id, trial);
    });

    let Some((distance, score, schedule)) = initial_state.into_inner().unwrap().best else {
        return Err("failed to build initial schedule".to_string());
    };
    Ok((schedule, score, distance))
}

fn sample_initial_search_weights(
    rng: &mut impl Random,
    state: &Mutex<InitialSearchState>,
) -> BlockOrderWeights {
    if rng.nextf() < INITIAL_EXPLOIT_PROB {
        let base = {
            let state = state.lock().unwrap();
            if state.good_weights.is_empty() {
                None
            } else {
                Some(state.good_weights[rng.gen_index(state.good_weights.len())].2)
            }
        };
        if let Some(base) = base {
            return mutate_initial_order_weights(base, rng);
        }
    }
    sample_initial_order_weights(rng)
}

fn update_good_weight_pool(
    pool: &mut Vec<(f64, f64, BlockOrderWeights)>,
    distance: f64,
    score: f64,
    weights: BlockOrderWeights,
) {
    if pool.len() < INITIAL_GOOD_WEIGHT_POOL_SIZE {
        pool.push((distance, score, weights));
        return;
    }

    let worst_idx = pool
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)))
        .map(|(idx, _)| idx)
        .unwrap();
    if distance
        .total_cmp(&pool[worst_idx].0)
        .then(score.total_cmp(&pool[worst_idx].1))
        .is_lt()
    {
        pool[worst_idx] = (distance, score, weights);
    }
}

fn build_initial_order<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    weights: BlockOrderWeights,
    rng: &mut R,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
    sort_block_order(problem, pre, &mut order, weights, rng);
    order
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

fn build_initial_schedule_with_order(
    problem: &Problem,
    pre: &Precompute,
    order: &[usize],
    guidance: &[PreoptimizedBlock],
) -> Option<Vec<ScheduledBlock>> {
    let mut schedule = Vec::with_capacity(problem.blocks.len());
    let mut loads = vec![0.0; problem.bays.len()];

    for &block_id in order {
        let block = &problem.blocks[block_id];
        let target = guidance[block_id];
        let original = ScheduledBlock {
            block_id,
            bay_id: target.bay_id,
            orient_idx: 0,
            x: 0,
            y: 0,
            entry_time: target.entry_time,
            exit_time: target.entry_time + block.processing_time,
        };
        let fixed_bay = [target.bay_id];
        let scheduled = insert_greedy(
            problem,
            pre,
            original,
            &schedule,
            &loads,
            INSERT_PARAMS,
            &fixed_bay,
            InsertMode::ClosestTo {
                target_time: target.entry_time,
            },
        )?;
        loads[scheduled.bay_id] += block.workload as f64;
        schedule.push(scheduled);
    }

    Some(schedule)
}

fn insert_mode(guidance: Option<&[PreoptimizedBlock]>, block_id: usize) -> InsertMode {
    guidance.map_or(InsertMode::Earliest, |guidance| InsertMode::ClosestTo {
        target_time: guidance[block_id].entry_time,
    })
}

fn try_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    score_limit: Option<f64>,
    guidance: Option<&[PreoptimizedBlock]>,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng).min(problem.blocks.len());

    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng, guidance);
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
        if score_limit.is_some_and(|score_limit| fixed_score13 > score_limit + 1e-9) {
            return None;
        }
        let old = old?;
        let fixed_bay = [old.bay_id];
        let bay_order = guidance.map_or(pre.bay_order_by_pref[old.block_id].as_slice(), |_| {
            fixed_bay.as_slice()
        });
        let scheduled = insert_greedy(
            problem,
            pre,
            old,
            &cur,
            &loads,
            INSERT_PARAMS,
            bay_order,
            insert_mode(guidance, old.block_id),
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
    mode: InsertMode,
) {
    let block = &problem.blocks[candidate.block_id];
    let tardiness = (candidate.exit_time - block.due_date).max(0);
    let better =
        best.as_ref().is_none_or(
            |&(best_tardiness, best_entry_time, best_scheduled)| match mode {
                InsertMode::Earliest => {
                    (tardiness, candidate.entry_time) < (best_tardiness, best_entry_time)
                }
                InsertMode::ClosestTo { target_time } => {
                    (
                        candidate.entry_time.abs_diff(target_time),
                        tardiness,
                        candidate.entry_time,
                    ) < (
                        best_scheduled.entry_time.abs_diff(target_time),
                        best_tardiness,
                        best_entry_time,
                    )
                }
            },
        );
    if better {
        *best = Some((tardiness, candidate.entry_time, candidate));
    }
}

fn try_shift_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    guidance: Option<&[PreoptimizedBlock]>,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];
    let mode = insert_mode(guidance, old.block_id);

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
                mode,
            ) else {
                continue;
            };
            if moved == old {
                continue;
            }
            update_best(problem, &mut best, moved, mode);
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
    guidance: Option<&[PreoptimizedBlock]>,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];
    let mode = insert_mode(guidance, old.block_id);
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
                    mode,
                ) else {
                    continue;
                };
                if rotated == old {
                    continue;
                }

                update_best(problem, &mut best, rotated, mode);
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
    mode: InsertMode,
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
                mode,
            ) else {
                continue;
            };
            update_best(problem, &mut best, scheduled, mode);
        }
    }

    Some(best?.2)
}

fn try_swap_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    guidance: Option<&[PreoptimizedBlock]>,
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
    if guidance.is_some() && a_old.bay_id != b_old.bay_id {
        return None;
    }

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
        insert_mode(guidance, a_old.block_id),
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
        insert_mode(guidance, b_old.block_id),
    )?;
    cur.push(b_new);

    Some(cur)
}

fn try_move_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    guidance: Option<&[PreoptimizedBlock]>,
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
        let sa = guidance.map_or_else(
            || move_obj13(problem, pre, schedule[a]),
            |guidance| {
                guidance_block_distance(
                    schedule[a],
                    guidance[schedule[a].block_id],
                    GUIDE_BAY_MISMATCH_PENALTY,
                )
            },
        );
        let sb = guidance.map_or_else(
            || move_obj13(problem, pre, schedule[b]),
            |guidance| {
                guidance_block_distance(
                    schedule[b],
                    guidance[schedule[b].block_id],
                    GUIDE_BAY_MISMATCH_PENALTY,
                )
            },
        );
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

    let fixed_bay = [old.bay_id];
    let bay_order = guidance.map_or(pre.bay_order_by_pref[old.block_id].as_slice(), |_| {
        fixed_bay.as_slice()
    });
    let scheduled = insert_greedy(
        problem,
        pre,
        old,
        &base,
        &loads,
        INSERT_PARAMS,
        bay_order,
        insert_mode(guidance, old.block_id),
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
    guidance: Option<&[PreoptimizedBlock]>,
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

    let mut badness = vec![0.0; problem.blocks.len()];
    for &s in schedule {
        badness[s.block_id] = guidance.map_or_else(
            || {
                let block = &problem.blocks[s.block_id];
                let tardiness = (s.exit_time - block.due_date).max(0);
                let pref_penalty = pre.pref_penalty[s.block_id][s.bay_id];
                tardiness as f64 * problem.weights.w1
                    + pref_penalty as f64 * problem.weights.w3
                    + if Some(s.bay_id) == heavy_bay {
                        problem.weights.w2
                    } else {
                        0.
                    }
            },
            |guidance| guidance_block_distance(s, guidance[s.block_id], GUIDE_BAY_MISMATCH_PENALTY),
        );
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
    ids.sort_by(|&a, &b| badness[b].total_cmp(&badness[a]).then(a.cmp(&b)));
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
