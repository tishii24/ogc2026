use crate::{
    collision::{BlockOrient, BlockPlacement, CollisionResult},
    precompute::Precompute,
    util::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
    vis::WorkerVisualizer,
    *,
};
use rayon::prelude::*;
use std::{cmp::Reverse, path::Path, sync::Mutex, time::Instant};

macro_rules! log {
    ($timer:expr, $($arg:tt)*) => {
        eprintln!("[{:.4}] {}", $timer.elapsed_seconds(), format_args!($($arg)*))
    };
}

const RNG_SEED: u64 = 1;
const MAX_WORKER_COUNT: usize = 4;

const LOCAL_SEARCH_TIME_BUFFER_SECONDS: f64 = 3.;
const START_TEMP: f64 = 1e1;
const END_TEMP: f64 = 1e0;

pub const PRECOMPUTE_ORIENTATION_NEIGHBOR_LIMIT: usize = 100;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_AREA_TOP_K: usize = 16;
pub const PRECOMPUTE_OTHER_BLOCK_NEIGHBOR_ALIGN_DELTA: i64 = 3;

const MAX_INITIAL_SEARCH_SECONDS: f64 = 30.;
const BASE_INITIAL_SEARCH_TIME_RATIO: f64 = 0.3;

const MIN_REMOVED_BLOCKS: usize = 1;
const MAX_REMOVED_BLOCKS: usize = 13;
const REMOVE_POOL_FACTOR: usize = 8;
const REMOVE_SEED_COUNT: usize = 3;
const REMOVE_RANDOM_SEED_COUNT: usize = 1;
const REMOVE_NEIGHBOR_POOL_FACTOR: usize = 4;
const INSERT_PARAMS: InsertSearchParams = InsertSearchParams { x_buffer: 10 };

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
    (NeighborKind::Move, 0.),
    (NeighborKind::Rotate, 8.),
    (NeighborKind::Swap, 3.),
];

const VISUALIZE_ACCEPTED_INTERVAL: usize = 1000;

fn sample_order_weights(rng: &mut impl Random) -> BlockOrderWeights {
    BlockOrderWeights {
        workload: rng.gen_rangef(0.0, 0.1),
        area: rng.gen_rangef(0.0, 0.2),
        pref_spread: rng.gen_rangef(0.0, 0.1),
        due_urgency: rng.gen_rangef(0.0, 1.0),
        slack_urgency: rng.gen_rangef(0.0, 1.0),
    }
}

type Interval = (i64, i64);

#[derive(Clone, Copy, Debug)]
enum NeighborKind {
    LargeReconstruct,
    Shift,
    Move,
    Rotate,
    Swap,
}

impl NeighborKind {
    fn name(&self) -> &'static str {
        match self {
            NeighborKind::LargeReconstruct => "Large",
            NeighborKind::Shift => "Shift",
            NeighborKind::Move => "Move",
            NeighborKind::Rotate => "Rotate",
            NeighborKind::Swap => "Swap",
        }
    }

    fn index(&self) -> usize {
        match self {
            NeighborKind::LargeReconstruct => 0,
            NeighborKind::Shift => 1,
            NeighborKind::Move => 2,
            NeighborKind::Rotate => 3,
            NeighborKind::Swap => 4,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct NeighborStats {
    selected: usize,
    succeeded: usize,
    improved: usize,
    accepted: usize,
    improved_delta_sum: f64,
    time_sec: f64,
}

#[derive(Clone, Copy)]
struct InsertSearchParams {
    x_buffer: i64,
}

#[derive(Clone, Copy)]
struct BlockOrderWeights {
    workload: f64,
    area: f64,
    pref_spread: f64,
    due_urgency: f64,
    slack_urgency: f64,
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

struct InsertCandidate {
    scheduled: ScheduledBlock,
    score_delta: f64,
    bbox_right: f64,
    bbox_top: f64,
}

#[derive(Clone, Copy)]
enum HitDir {
    NewOld,
    OldNew,
}

#[derive(Clone, Copy)]
struct YEvent {
    y: i64,
    old_idx: usize,
    dir: HitDir,
    delta: i8,
}

#[derive(Clone, Copy, Default)]
struct HitState {
    new_old: u16,
    old_new: u16,
}

impl HitState {
    fn is_active(self) -> bool {
        self.new_old > 0 || self.old_new > 0
    }
}

#[derive(Clone, Copy)]
struct OldTimeInfo {
    old_entry_time: i64,
    old_exit_time: i64,
    new_process_time: i64,
    overlap_entry_time_min: i64,
    overlap_entry_time_max: i64,
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

pub fn solve(
    problem: &Problem,
    timelimit: f64,
    timer: Timer,
    visualize_dir: Option<&Path>,
) -> Result<Solution, String> {
    log!(timer, "building precompute...");
    let pre = Precompute::build(problem);
    log!(timer, "precompute built");

    let deadline = timelimit - LOCAL_SEARCH_TIME_BUFFER_SECONDS;
    let worker_count = rayon::current_num_threads().clamp(1, MAX_WORKER_COUNT);
    log!(timer, "annealing workers: {}", worker_count);

    let (initial, initial_score) =
        search_initial_schedule(problem, &pre, timelimit, timer, worker_count)?;
    log!(timer, "best initial score: {:.3}", initial_score);

    let worker_timers = vec![timer; worker_count];
    let results: Vec<_> = worker_timers
        .into_par_iter()
        .enumerate()
        .map(|(worker_id, worker_timer)| {
            run_annealing_worker(
                problem,
                &pre,
                &initial,
                deadline,
                worker_id,
                worker_timer,
                visualize_dir,
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
            format_neighbor_stats(&result.neighbor_stats),
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
    initial: &[ScheduledBlock],
    deadline: f64,
    worker_id: usize,
    timer: Timer,
    visualize_dir: Option<&Path>,
) -> AnnealingResult {
    let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(worker_id as u64));
    let mut current = initial.to_vec();
    let mut current_score = score_schedule(problem, pre, &current);
    let mut best = current.clone();
    let mut best_score = current_score;
    let mut iter = 0usize;
    let mut accepted = 0usize;
    let mut improved = 0usize;
    let mut neighbor_stats = [NeighborStats::default(); NEIGHBOR_KIND_COUNT];
    let mut visualizer =
        visualize_dir.and_then(|dir| match WorkerVisualizer::create(dir, worker_id) {
            Ok(visualizer) => Some(visualizer),
            Err(err) => {
                eprintln!("failed to create visualizer for worker {worker_id}: {err}");
                None
            }
        });
    if let Some(visualizer) = visualizer.as_mut() {
        if let Err(err) = visualizer.write_snapshot(
            "initial",
            worker_id,
            0,
            timer.elapsed_seconds(),
            None,
            None,
            false,
            false,
            current_score,
            current_score,
            best_score,
            None,
            &current,
        ) {
            eprintln!("failed to write visualizer snapshot for worker {worker_id}: {err}");
        }
    }

    loop {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= deadline {
            break;
        }
        iter += 1;
        let progress = (elapsed / deadline).clamp(0.0, 1.0);
        let temp = START_TEMP * (END_TEMP / START_TEMP).powf(progress);

        let neighbor = sample_neighbor(&mut rng);
        let neighbor_idx = neighbor.index();
        let neighbor_start = Instant::now();
        neighbor_stats[neighbor_idx].selected += 1;

        let candidate = match neighbor {
            NeighborKind::LargeReconstruct => {
                try_large_reconstruct(problem, pre, &current, &mut rng)
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

        let score = score_schedule(problem, pre, &candidate);
        let delta = score - current_score;
        let improved_current = delta < -1e-9;
        if improved_current {
            neighbor_stats[neighbor_idx].improved += 1;
            neighbor_stats[neighbor_idx].improved_delta_sum += -delta;
        }
        if delta <= 0.0 || rng.nextf() < (-delta / temp).exp() {
            current = candidate;
            current_score = score;
            accepted += 1;
            neighbor_stats[neighbor_idx].accepted += 1;

            let mut improved_best = false;
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
                improved_best = true;
            }

            let reason = if improved_best {
                Some("best")
            } else if accepted % VISUALIZE_ACCEPTED_INTERVAL == 0 {
                Some("periodic")
            } else {
                None
            };
            if let (Some(reason), Some(visualizer)) = (reason, visualizer.as_mut()) {
                if let Err(err) = visualizer.write_snapshot(
                    reason,
                    worker_id,
                    iter,
                    elapsed,
                    Some(neighbor.name()),
                    Some(true),
                    improved_current,
                    improved_best,
                    score,
                    current_score,
                    best_score,
                    Some(delta),
                    &current,
                ) {
                    eprintln!("failed to write visualizer snapshot for worker {worker_id}: {err}");
                }
            }
        }
        neighbor_stats[neighbor_idx].time_sec += neighbor_start.elapsed().as_secs_f64();
    }

    if let Some(visualizer) = visualizer.as_mut() {
        if let Err(err) = visualizer.write_snapshot(
            "final",
            worker_id,
            iter,
            timer.elapsed_seconds(),
            None,
            None,
            false,
            false,
            best_score,
            current_score,
            best_score,
            None,
            &best,
        ) {
            eprintln!("failed to write visualizer snapshot for worker {worker_id}: {err}");
        }
        if let Err(err) = visualizer.flush() {
            eprintln!("failed to flush visualizer for worker {worker_id}: {err}");
        }
    }

    AnnealingResult {
        worker_id,
        schedule: best,
        score: best_score,
        current_score,
        iter,
        accepted,
        improved,
        neighbor_stats,
    }
}

fn search_initial_schedule(
    problem: &Problem,
    pre: &Precompute,
    timelimit: f64,
    timer: Timer,
    worker_count: usize,
) -> Result<(Vec<ScheduledBlock>, f64), String> {
    let search_seconds = MAX_INITIAL_SEARCH_SECONDS.min(BASE_INITIAL_SEARCH_TIME_RATIO * timelimit);
    let search_deadline = timer.elapsed_seconds() + search_seconds;
    log!(timer, "initial search seconds: {:.3}", search_seconds);

    let best = Mutex::new(None::<(f64, Vec<ScheduledBlock>)>);
    (0..worker_count).into_par_iter().for_each(|worker_id| {
        let mut trial = 0;
        let mut rng =
            RandPcg64Mcg::new(RNG_SEED.wrapping_add(10_000).wrapping_add(worker_id as u64));

        loop {
            if timer.elapsed_seconds() >= search_deadline && best.lock().unwrap().is_some() {
                break;
            }

            let weights = sample_order_weights(&mut rng);
            let Ok(schedule) = build_initial_schedule(problem, pre, weights) else {
                continue;
            };
            let score = score_schedule(problem, pre, &schedule);
            let mut best_guard = best.lock().unwrap();
            if best_guard
                .as_ref()
                .map_or(true, |(best_score, _)| score + 1e-9 < *best_score)
            {
                log!(
                    timer,
                    "update initial-score: worker {} score: {:.3}",
                    worker_id,
                    score
                );
                *best_guard = Some((score, schedule));
            }

            trial += 1;
        }

        log!(timer, "worker {} trial {:4}", worker_id, trial);
    });

    let Some((score, schedule)) = best.into_inner().unwrap() else {
        return Err("failed to build initial schedule".to_string());
    };
    Ok((schedule, score))
}

fn build_initial_schedule(
    problem: &Problem,
    pre: &Precompute,
    weights: BlockOrderWeights,
) -> Result<Vec<ScheduledBlock>, String> {
    let mut schedule = Vec::with_capacity(problem.blocks.len());
    let mut loads = vec![0.0; problem.bays.len()];
    let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
    sort_block_order(problem, pre, &mut order, weights);

    for block_id in order {
        let block = &problem.blocks[block_id];
        let original = ScheduledBlock {
            block_id,
            bay_id: 0,
            orient_idx: 0,
            x: 0,
            y: 0,
            entry_time: block.release_time,
            exit_time: block.release_time + block.processing_time,
        };
        let scheduled = insert_greedy(problem, pre, original, &schedule, &loads, INSERT_PARAMS)
            .ok_or_else(|| format!("failed to place block {block_id} in initial schedule"))?;
        loads[scheduled.bay_id] += block.workload as f64;
        schedule.push(scheduled);
    }

    Ok(schedule)
}

fn try_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
) -> Option<Vec<ScheduledBlock>> {
    let k = rng
        .gen_range(MIN_REMOVED_BLOCKS, MAX_REMOVED_BLOCKS + 1)
        .min(problem.blocks.len());

    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng);
    if removed_ids.is_empty() {
        return None;
    }
    let w = sample_order_weights(rng);
    sort_block_order(problem, pre, &mut removed_ids, w);

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
    for s in &cur {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }

    for old in removed_ordered {
        let old = old?;
        let scheduled = insert_greedy(problem, pre, old, &cur, &loads, INSERT_PARAMS)?;
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        cur.push(scheduled);
    }

    Some(cur)
}

fn try_place_block(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    block_id: usize,
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
) -> Option<ScheduledBlock> {
    let block = &problem.blocks[block_id];
    let tentative = ScheduledBlock {
        block_id,
        bay_id,
        orient_idx,
        x,
        y,
        entry_time: 0,
        exit_time: block.processing_time,
    };
    let entry_time = get_insert_t(pre, tentative, schedule, block.release_time, i64::MAX)?;
    Some(ScheduledBlock {
        entry_time,
        exit_time: entry_time + block.processing_time,
        ..tentative
    })
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

    let scheduled = insert_greedy(problem, pre, old, &base, &loads, INSERT_PARAMS)?;
    if scheduled == old {
        return None;
    }

    base.push(scheduled);
    Some(base)
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
    let seed_count = REMOVE_SEED_COUNT.min(k);
    let random_seed_count = REMOVE_RANDOM_SEED_COUNT.min(seed_count);
    let bad_seed_count = seed_count - random_seed_count;

    let mut seed_pool = bad_pool.clone();
    rng.shuffle(&mut seed_pool);
    for &block_id in seed_pool.iter().take(bad_seed_count) {
        push_selected(&mut selected, &mut used, block_id, k);
    }

    let mut random_seed_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut random_seed_pool);
    for block_id in random_seed_pool {
        if selected.len() >= seed_count {
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
        let mut neighbors: Vec<(f64, usize)> = schedule
            .iter()
            .filter(|s| s.bay_id == seed.bay_id && !used[s.block_id])
            .map(|&s| {
                let (x, y) = scheduled_center(pre, s);
                let dx = x - sx;
                let dy = y - sy;
                (dx * dx + dy * dy, s.block_id)
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let pool_len = (need * REMOVE_NEIGHBOR_POOL_FACTOR).min(neighbors.len());
        let mut neighbor_ids: Vec<usize> = neighbors
            .into_iter()
            .take(pool_len)
            .map(|(_, block_id)| block_id)
            .collect();
        rng.shuffle(&mut neighbor_ids);
        for block_id in neighbor_ids.into_iter().take(need) {
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

fn sort_block_order(
    problem: &Problem,
    pre: &Precompute,
    order: &mut [usize],
    weights: BlockOrderWeights,
) {
    let ctx = build_block_order_context(problem, pre, order);
    order.sort_by(|&a, &b| {
        let score_a = block_order_score(problem, pre, &ctx, weights, a);
        let score_b = block_order_score(problem, pre, &ctx, weights, b);
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

fn insert_greedy(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    schedule: &[ScheduledBlock],
    loads: &[f64],
    params: InsertSearchParams,
) -> Option<ScheduledBlock> {
    fn insert_candidate_better(a: &InsertCandidate, b: &InsertCandidate) -> bool {
        a.score_delta
            .total_cmp(&b.score_delta)
            .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
            .then(a.bbox_right.total_cmp(&b.bbox_right))
            .then(a.bbox_top.total_cmp(&b.bbox_top))
            .then(a.scheduled.block_id.cmp(&b.scheduled.block_id))
            .is_lt()
    }

    let block_id = original.block_id;
    let block = &problem.blocks[block_id];
    let process_t = block.processing_time;
    let min_t = block.release_time;
    let max_t = i64::MAX;

    let current_obj2 = normalized_imbalance(pre, loads);
    let original_tardiness = (original.exit_time - block.due_date).max(0);
    let mut best: Option<InsertCandidate> = None;

    for bay_id in 0..problem.bays.len() {
        let bay_old_blocks: Vec<ScheduledBlock> = schedule
            .iter()
            .copied()
            .filter(|old| old.bay_id == bay_id)
            .collect();
        let old_time_infos: Vec<Option<OldTimeInfo>> = bay_old_blocks
            .iter()
            .map(|&old| old_time_info(old, process_t, min_t, max_t))
            .collect();
        let mut events = Vec::with_capacity(bay_old_blocks.len() * 4);
        let mut states = vec![HitState::default(); bay_old_blocks.len()];
        let mut active_old_ids = Vec::with_capacity(bay_old_blocks.len());
        let mut active_pos = vec![None; bay_old_blocks.len()];
        let mut forbidden = Vec::with_capacity(16);

        let mut next_loads = loads.to_vec();
        next_loads[bay_id] += block.workload as f64;
        let delta_obj23 = problem.weights.w2
            * (normalized_imbalance(pre, &next_loads) - current_obj2)
            + problem.weights.w3 * pre.pref_penalty[block_id][bay_id] as f64;

        for &orient_idx in &pre.orientation_order_by_bbox[block_id] {
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            let new_orient = BlockOrient {
                block_id,
                orient_idx,
            };
            let bounds = pre.orientation_bbox_bounds[block_id][orient_idx];
            let mut anchor_x: Option<i64> = None;

            for x in range.min_x..=range.max_x {
                if let Some(anchor_x) = anchor_x {
                    if x > anchor_x + params.x_buffer {
                        break;
                    }
                }

                events.clear();
                states.fill(HitState::default());
                active_old_ids.clear();
                active_pos.fill(None);

                for (old_idx, &old) in bay_old_blocks.iter().enumerate() {
                    if old_time_infos[old_idx].is_none() {
                        continue;
                    }

                    let old_orient = BlockOrient {
                        block_id: old.block_id,
                        orient_idx: old.orient_idx,
                    };

                    for &(lo, hi) in
                        pre.collision
                            .crane_dy_intervals(new_orient, old_orient, old.x - x)
                    {
                        push_y_event(
                            old.y - hi,
                            old.y - lo,
                            old_idx,
                            HitDir::NewOld,
                            range.min_y,
                            range.max_y,
                            &mut events,
                        );
                    }
                    for &(lo, hi) in
                        pre.collision
                            .crane_dy_intervals(old_orient, new_orient, x - old.x)
                    {
                        push_y_event(
                            old.y + lo,
                            old.y + hi,
                            old_idx,
                            HitDir::OldNew,
                            range.min_y,
                            range.max_y,
                            &mut events,
                        );
                    }
                }

                events.sort_unstable_by_key(|event| event.y);
                let mut event_pos = 0;
                let mut y = range.min_y;
                loop {
                    while event_pos < events.len() && events[event_pos].y == y {
                        apply_y_event(
                            events[event_pos],
                            &mut states,
                            &mut active_old_ids,
                            &mut active_pos,
                        );
                        event_pos += 1;
                    }

                    forbidden.clear();
                    for &old_idx in &active_old_ids {
                        let Some(info) = old_time_infos[old_idx] else {
                            continue;
                        };
                        let state = states[old_idx];
                        add_forbidden_from_hit_state(
                            info,
                            state.new_old > 0,
                            state.old_new > 0,
                            &mut forbidden,
                        );
                    }

                    if let Some(entry_time) = first_feasible_time(&forbidden, min_t, max_t) {
                        let scheduled = ScheduledBlock {
                            block_id,
                            bay_id,
                            orient_idx,
                            x,
                            y,
                            entry_time,
                            exit_time: entry_time + process_t,
                        };
                        let tardiness = (scheduled.exit_time - block.due_date).max(0);
                        let score_delta = problem.weights.w1 * tardiness as f64 + delta_obj23;
                        let candidate = InsertCandidate {
                            scheduled,
                            score_delta,
                            bbox_right: scheduled.x as f64 + bounds.max_x,
                            bbox_top: scheduled.y as f64 + bounds.max_y,
                        };
                        if best
                            .as_ref()
                            .map_or(true, |best| insert_candidate_better(&candidate, best))
                        {
                            best = Some(candidate);
                        }

                        if tardiness <= original_tardiness && anchor_x.is_none() {
                            anchor_x = Some(x);
                        }
                    }

                    if event_pos >= events.len() {
                        break;
                    }
                    y = events[event_pos].y;
                    if y > range.max_y {
                        break;
                    }
                }
            }
        }
    }

    best.map(|candidate| candidate.scheduled)
}

fn old_time_info(
    old: ScheduledBlock,
    process_t: i64,
    min_t: i64,
    max_t: i64,
) -> Option<OldTimeInfo> {
    let a = old.entry_time;
    let b = old.exit_time;
    let ol = (a - process_t + 1).max(min_t);
    let or = (b - 1).min(max_t);
    if ol <= or {
        Some(OldTimeInfo {
            old_entry_time: a,
            old_exit_time: b,
            new_process_time: process_t,
            overlap_entry_time_min: ol,
            overlap_entry_time_max: or,
        })
    } else {
        None
    }
}

fn push_y_event(
    l: i64,
    r: i64,
    old_idx: usize,
    dir: HitDir,
    min_y: i64,
    max_y: i64,
    events: &mut Vec<YEvent>,
) {
    let l = l.max(min_y);
    let r = r.min(max_y);
    if l > r {
        return;
    }

    events.push(YEvent {
        y: l,
        old_idx,
        dir,
        delta: 1,
    });
    if let Some(y) = r.checked_add(1) {
        events.push(YEvent {
            y,
            old_idx,
            dir,
            delta: -1,
        });
    }
}

fn apply_y_event(
    event: YEvent,
    states: &mut [HitState],
    active_old_ids: &mut Vec<usize>,
    active_pos: &mut [Option<usize>],
) {
    let old_idx = event.old_idx;
    let was_active = states[old_idx].is_active();

    match event.dir {
        HitDir::NewOld => {
            if event.delta > 0 {
                states[old_idx].new_old += 1;
            } else {
                debug_assert!(states[old_idx].new_old > 0);
                states[old_idx].new_old -= 1;
            }
        }
        HitDir::OldNew => {
            if event.delta > 0 {
                states[old_idx].old_new += 1;
            } else {
                debug_assert!(states[old_idx].old_new > 0);
                states[old_idx].old_new -= 1;
            }
        }
    }

    let is_active = states[old_idx].is_active();
    if !was_active && is_active {
        active_pos[old_idx] = Some(active_old_ids.len());
        active_old_ids.push(old_idx);
    } else if was_active && !is_active {
        let pos = active_pos[old_idx].take().unwrap();
        let last = active_old_ids.pop().unwrap();
        if pos < active_old_ids.len() {
            active_old_ids[pos] = last;
            active_pos[last] = Some(pos);
        }
    }
}

fn add_forbidden_from_hit_state(
    info: OldTimeInfo,
    new_old_hit: bool,
    old_new_hit: bool,
    forbidden: &mut Vec<Interval>,
) {
    let new_old_clear = !new_old_hit;
    let old_new_clear = !old_new_hit;
    if new_old_clear && old_new_clear {
        return;
    }

    let ol = info.overlap_entry_time_min;
    let or = info.overlap_entry_time_max;
    if !new_old_clear && !old_new_clear {
        forbidden.push((ol, or));
        return;
    }

    let (allow_l, allow_r) = if new_old_clear {
        (
            (info.old_entry_time + 1).max(ol),
            (info.old_exit_time - info.new_process_time - 1).min(or),
        )
    } else {
        (
            (info.old_exit_time - info.new_process_time + 1).max(ol),
            (info.old_entry_time - 1).min(or),
        )
    };

    if allow_l > allow_r {
        forbidden.push((ol, or));
        return;
    }
    if ol < allow_l {
        forbidden.push((ol, allow_l - 1));
    }
    if allow_r < or {
        forbidden.push((allow_r + 1, or));
    }
}

/// TODO: sort版も試す
fn first_feasible_time(forbidden: &[Interval], min_t: i64, max_t: i64) -> Option<i64> {
    let mut t = min_t;
    loop {
        let mut next_t = t;
        for &(l, r) in forbidden {
            if l <= t && t <= r {
                if r == i64::MAX {
                    return None;
                }
                next_t = next_t.max(r + 1);
            }
        }

        if next_t == t {
            return Some(t);
        }
        if next_t > max_t {
            return None;
        }
        t = next_t;
    }
}

fn get_insert_t(
    pre: &Precompute,
    new_block: ScheduledBlock,
    schedule: &[ScheduledBlock],
    min_t: i64,
    max_t: i64,
) -> Option<i64> {
    let mut forbidden = Vec::with_capacity(16);
    for &old in schedule.iter().filter(|old| old.bay_id == new_block.bay_id) {
        let process_t = new_block.exit_time - new_block.entry_time;
        let Some(info) = old_time_info(old, process_t, min_t, max_t) else {
            continue;
        };

        let new_place = BlockPlacement {
            block_id: new_block.block_id,
            orient_idx: new_block.orient_idx,
            x: new_block.x,
            y: new_block.y,
        };
        let old_place = BlockPlacement {
            block_id: old.block_id,
            orient_idx: old.orient_idx,
            x: old.x,
            y: old.y,
        };

        let new_old_hit = pre.collision.crane(new_place, old_place) == CollisionResult::Hit;
        let old_new_hit = pre.collision.crane(old_place, new_place) == CollisionResult::Hit;
        add_forbidden_from_hit_state(info, new_old_hit, old_new_hit, &mut forbidden);
    }
    first_feasible_time(&forbidden, min_t, max_t)
}

fn score_schedule(problem: &Problem, pre: &Precompute, schedule: &[ScheduledBlock]) -> f64 {
    let mut obj1 = 0.0;
    let mut obj3 = 0.0;
    let mut loads = vec![0.0; problem.bays.len()];

    for s in schedule {
        let block = &problem.blocks[s.block_id];
        obj1 += (s.exit_time - block.due_date).max(0) as f64;
        loads[s.bay_id] += block.workload as f64;
        obj3 += pre.pref_penalty[s.block_id][s.bay_id] as f64;
    }

    let obj2 = normalized_imbalance(pre, &loads);
    problem.weights.w1 * obj1 + problem.weights.w2 * obj2 + problem.weights.w3 * obj3
}

fn normalized_imbalance(pre: &Precompute, loads: &[f64]) -> f64 {
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

fn schedule_to_solution(schedule: &[ScheduledBlock]) -> Solution {
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

fn sample_neighbor<R: Random>(rng: &mut R) -> NeighborKind {
    let total = NEIGHBOR_PROBS.iter().map(|&(_, prob)| prob).sum::<f64>();
    debug_assert!(total > 0.0);

    let mut x = rng.nextf() * total;
    for &(kind, prob) in NEIGHBOR_PROBS {
        if x < prob {
            return kind;
        }
        x -= prob;
    }
    NEIGHBOR_PROBS.last().unwrap().0
}

fn format_neighbor_stats(stats: &[NeighborStats; NEIGHBOR_KIND_COUNT]) -> String {
    fn ratio(num: usize, den: usize) -> f64 {
        if den == 0 {
            0.0
        } else {
            100.0 * num as f64 / den as f64
        }
    }

    fn avg_ms(total_sec: f64, count: usize) -> f64 {
        if count == 0 {
            0.0
        } else {
            total_sec * 1000.0 / count as f64
        }
    }

    fn avg_sum(sum: f64, count: usize) -> f64 {
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    NEIGHBOR_PROBS
        .iter()
        .map(|&(kind, _)| {
            let stat = stats[kind.index()];
            format!(
                "  {:<8}: selected={:7}, succeeded={:7} ({:7.3}%), improved={:7} ({:7.3}%, avg={:10.2}), accepted={:7} ({:7.3}%), time={:7.3}s, avg={:7.3}ms",
                kind.name(),
                stat.selected,
                stat.succeeded,
                ratio(stat.succeeded, stat.selected),
                stat.improved,
                ratio(stat.improved, stat.succeeded),
                avg_sum(stat.improved_delta_sum, stat.improved),
                stat.accepted,
                ratio(stat.accepted, stat.succeeded),
                stat.time_sec,
                avg_ms(stat.time_sec, stat.selected),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
