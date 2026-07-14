use super::*;
use crate::{
    solver_util::{NeighborStats, format_neighbor_stats, sample_neighbor},
    tabu::ScheduleTabu,
};
use rayon::prelude::*;
use std::{sync::Mutex, time::Instant};

#[derive(Default)]
struct AnnealingStats {
    iter: usize,
    accepted: usize,
    improved: usize,
    neighbor_stats: [NeighborStats; NEIGHBOR_KIND_COUNT],
}

struct ScheduleAnnealingState {
    current: Vec<ScheduledBlock>,
    current_score: f64,
    best: Vec<ScheduledBlock>,
    best_score: f64,
    best_metric: f64,
    stats: AnnealingStats,
}

struct AcceptOutcome {
    accepted: bool,
    new_best: bool,
}

impl ScheduleAnnealingState {
    fn new(initial: Vec<ScheduledBlock>, score: f64, metric: f64) -> Self {
        Self {
            current: initial.clone(),
            current_score: score,
            best: initial,
            best_score: score,
            best_metric: metric,
            stats: AnnealingStats::default(),
        }
    }

    fn accept_candidate(
        &mut self,
        candidate: Vec<ScheduledBlock>,
        score: f64,
        metric: f64,
        accept_threshold: f64,
        neighbor_idx: usize,
    ) -> AcceptOutcome {
        let delta = score - self.current_score;
        if delta < -1e-9 {
            self.stats.neighbor_stats[neighbor_idx].improved += 1;
            self.stats.neighbor_stats[neighbor_idx].improved_delta_sum += -delta;
        }
        if score > accept_threshold {
            return AcceptOutcome {
                accepted: false,
                new_best: false,
            };
        }

        self.current = candidate;
        self.current_score = score;
        self.stats.accepted += 1;
        self.stats.neighbor_stats[neighbor_idx].accepted += 1;

        let new_best = metric + 1e-9 < self.best_metric;
        if new_best {
            self.best = self.current.clone();
            self.best_score = score;
            self.best_metric = metric;
            self.stats.improved += 1;
        }
        AcceptOutcome {
            accepted: true,
            new_best,
        }
    }

    fn adopt_external(&mut self, state: &OptimizeState) {
        self.current = state.blocks.clone();
        self.current_score = state.score;
        if state.score + 1e-9 < self.best_metric {
            self.best = state.blocks.clone();
            self.best_score = state.score;
            self.best_metric = state.score;
        }
    }
}

struct TemperatureSchedule {
    start_time: f64,
    deadline: f64,
    start_temperature: f64,
    end_temperature: f64,
}

impl TemperatureSchedule {
    fn new(start_time: f64, deadline: f64, start_temperature: f64, end_temperature: f64) -> Self {
        Self {
            start_time,
            deadline,
            start_temperature,
            end_temperature,
        }
    }

    fn temperature(&self, elapsed: f64) -> f64 {
        let progress = ((elapsed - self.start_time) / (self.deadline - self.start_time).max(1e-4))
            .clamp(0.0, 1.0);
        self.start_temperature * (self.end_temperature / self.start_temperature).powf(progress)
    }
}

struct AnnealingResult {
    worker_id: usize,
    state: OptimizeState,
    current_score: f64,
    stats: AnnealingStats,
}

struct BayAnnealingResult {
    bay_id: usize,
    blocks: Vec<ScheduledBlock>,
    tardiness: i64,
    target_tardiness: i64,
    stats: AnnealingStats,
}

impl BayAnnealingResult {
    fn idle(
        problem: &Problem,
        bay_id: usize,
        blocks: Vec<ScheduledBlock>,
        target_tardiness: i64,
    ) -> Self {
        let tardiness = bay_tardiness(problem, &blocks);
        Self {
            bay_id,
            blocks,
            tardiness,
            target_tardiness,
            stats: AnnealingStats::default(),
        }
    }
}

struct BayTask {
    bay_id: usize,
    blocks: Vec<ScheduledBlock>,
    target_tardiness: i64,
    result: Option<BayAnnealingResult>,
}

pub(super) struct BayAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: PrecedenceConstraints,
    target_tardiness: Vec<i64>,
    timer: Timer,
}

impl<'a> BayAnnealing<'a> {
    pub(super) fn new(
        problem: &'a Problem,
        pre: &'a Precompute,
        abstract_state: &PreoptimizeState,
        timer: Timer,
    ) -> Self {
        let constraints = build_precedence_constraints(problem, abstract_state);
        let mut target_tardiness = vec![0i64; problem.bays.len()];
        for (block_id, scheduled) in abstract_state.blocks.iter().enumerate() {
            let block = &problem.blocks[block_id];
            let exit_time = scheduled.entry_time + block.processing_time;
            target_tardiness[scheduled.bay_id] += (exit_time - block.due_date).max(0);
        }
        Self {
            problem,
            pre,
            constraints,
            target_tardiness,
            timer,
        }
    }

    pub(super) fn run(&self, initial: OptimizeState, deadline: f64) -> OptimizeState {
        if self.timer.elapsed_seconds() >= deadline {
            return initial;
        }

        let mut blocks_by_bay = vec![Vec::new(); self.problem.bays.len()];
        for block in initial.blocks {
            blocks_by_bay[block.bay_id].push(block);
        }

        let mut results: Vec<Option<BayAnnealingResult>> =
            (0..self.problem.bays.len()).map(|_| None).collect();
        let mut active = Vec::new();
        for (bay_id, blocks) in blocks_by_bay.into_iter().enumerate() {
            let target = self.target_tardiness[bay_id];
            if bay_tardiness(self.problem, &blocks) <= target {
                results[bay_id] = Some(BayAnnealingResult::idle(
                    self.problem,
                    bay_id,
                    blocks,
                    target,
                ));
            } else {
                active.push(BayTask {
                    bay_id,
                    blocks,
                    target_tardiness: target,
                    result: None,
                });
            }
        }

        if !active.is_empty() {
            let worker_count = rayon::current_num_threads()
                .clamp(1, MAX_WORKER_COUNT)
                .min(active.len());
            let batch_count = active.len().div_ceil(worker_count);
            for batch_index in 0..batch_count {
                let elapsed = self.timer.elapsed_seconds();
                let remaining_batches = batch_count - batch_index;
                let batch_deadline =
                    elapsed + (deadline - elapsed).max(0.0) / remaining_batches as f64;
                let begin = batch_index * worker_count;
                let end = (begin + worker_count).min(active.len());
                active[begin..end].par_iter_mut().for_each(|task| {
                    task.result = Some(self.run_single(
                        std::mem::take(&mut task.blocks),
                        task.target_tardiness,
                        batch_deadline,
                        task.bay_id,
                    ));
                });
            }
        }

        for task in active {
            results[task.bay_id] = task.result;
        }

        let mut blocks = Vec::with_capacity(self.problem.blocks.len());
        for result in results.into_iter().flatten() {
            eprintln!(
                "[{:.4}] [bay={}] iter={}, best_tardiness={}, target={}, accepted={}, improved={}\nneighbor stats:\n{}",
                self.timer.elapsed_seconds(),
                result.bay_id,
                result.stats.iter,
                result.tardiness,
                result.target_tardiness,
                result.stats.accepted,
                result.stats.improved,
                format_neighbor_stats(&result.stats.neighbor_stats, BAY_NEIGHBOR_PROBS),
            );
            blocks.extend(result.blocks);
        }
        let score = score_schedule(self.problem, self.pre, &blocks);
        OptimizeState { score, blocks }
    }

    fn run_single(
        &self,
        initial: Vec<ScheduledBlock>,
        target_tardiness: i64,
        deadline: f64,
        bay_id: usize,
    ) -> BayAnnealingResult {
        let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(10_000).wrapping_add(bay_id as u64));
        let initial_tardiness = bay_tardiness(self.problem, &initial);
        let initial_score = self.problem.weights.w1 * initial_tardiness as f64;
        let mut state =
            ScheduleAnnealingState::new(initial, initial_score, initial_tardiness as f64);
        let start = self.timer.elapsed_seconds();
        let start_temperature = (self.problem.weights.w1 / TEMP_WEIGHT_DIVISOR).max(1e-9);
        let temperature = TemperatureSchedule::new(
            start,
            deadline,
            start_temperature,
            (start_temperature * BAY_END_TEMPERATURE_RATIO).max(1e-9),
        );

        while state.best_metric > target_tardiness as f64 {
            let elapsed = self.timer.elapsed_seconds();
            if elapsed >= deadline {
                break;
            }
            state.stats.iter += 1;
            let temp = temperature.temperature(elapsed);
            let accept_threshold = state.current_score - temp * rng.nextf().ln();
            let neighbor = sample_neighbor(&mut rng, BAY_NEIGHBOR_PROBS);
            let neighbor_idx = neighbor.index();
            let neighbor_start = Instant::now();
            state.stats.neighbor_stats[neighbor_idx].selected += 1;

            let candidate = match neighbor {
                NeighborKind::LargeReconstruct => try_bay_large_reconstruct(
                    self.problem,
                    self.pre,
                    &self.constraints,
                    &state.current,
                    &mut rng,
                    accept_threshold,
                    bay_id,
                ),
                NeighborKind::Shift => try_shift_neighbor(
                    self.problem,
                    self.pre,
                    &state.current,
                    &mut rng,
                    Some(&self.constraints),
                ),
                NeighborKind::Move => try_move_neighbor(
                    self.problem,
                    self.pre,
                    &state.current,
                    &mut rng,
                    Some(&self.constraints),
                    Some(bay_id),
                ),
                NeighborKind::Rotate => try_rotate_neighbor(
                    self.problem,
                    self.pre,
                    &state.current,
                    &mut rng,
                    Some(&self.constraints),
                ),
                NeighborKind::Swap => None,
            };
            let Some(candidate) = candidate else {
                state.stats.neighbor_stats[neighbor_idx].time_sec +=
                    neighbor_start.elapsed().as_secs_f64();
                continue;
            };
            state.stats.neighbor_stats[neighbor_idx].succeeded += 1;

            let candidate_tardiness = bay_tardiness(self.problem, &candidate);
            let score = self.problem.weights.w1 * candidate_tardiness as f64;
            state.accept_candidate(
                candidate,
                score,
                candidate_tardiness as f64,
                accept_threshold,
                neighbor_idx,
            );
            state.stats.neighbor_stats[neighbor_idx].time_sec +=
                neighbor_start.elapsed().as_secs_f64();
        }

        BayAnnealingResult {
            bay_id,
            blocks: state.best,
            tardiness: state.best_metric as i64,
            target_tardiness,
            stats: state.stats,
        }
    }
}

pub(super) struct GlobalAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    timer: Timer,
}

impl<'a> GlobalAnnealing<'a> {
    pub(super) fn new(problem: &'a Problem, pre: &'a Precompute, timer: Timer) -> Self {
        Self {
            problem,
            pre,
            timer,
        }
    }

    pub(super) fn run(&self, initial: OptimizeState, deadline: f64) -> OptimizeState {
        let worker_count = rayon::current_num_threads().clamp(1, MAX_WORKER_COUNT);
        log!(self.timer, "annealing workers: {}", worker_count);
        let shared = Mutex::new(initial.clone());
        let results: Vec<_> = (0..worker_count)
            .into_par_iter()
            .map(|worker_id| self.run_worker(&shared, &initial, deadline, worker_id, worker_count))
            .collect();

        let mut best = initial;
        for result in results {
            eprintln!(
                "[{:.4}] [id={}] iter={}, best={:.3}, accepted={}, improved={}, current={:.3}\nneighbor stats:\n{}",
                self.timer.elapsed_seconds(),
                result.worker_id,
                result.stats.iter,
                result.state.score,
                result.stats.accepted,
                result.stats.improved,
                result.current_score,
                format_neighbor_stats(&result.stats.neighbor_stats, NEIGHBOR_PROBS),
            );
            if result.state.score + 1e-9 < best.score {
                best = result.state;
            }
        }
        best
    }

    fn run_worker(
        &self,
        shared: &Mutex<OptimizeState>,
        initial: &OptimizeState,
        deadline: f64,
        worker_id: usize,
        worker_count: usize,
    ) -> AnnealingResult {
        let mut rng = RandPcg64Mcg::new(RNG_SEED.wrapping_add(worker_id as u64));
        let temp_scale = get_temp_scale(worker_id, worker_count);
        let mut state =
            ScheduleAnnealingState::new(initial.blocks.clone(), initial.score, initial.score);
        let mut tabu = ScheduleTabu::new(TABU_SIZE);
        tabu.insert_schedule(&state.current);
        let start = self.timer.elapsed_seconds();
        let start_temperature =
            temp_scale * (self.problem.weights.w1 / TEMP_WEIGHT_DIVISOR).max(1e-9);
        let end_temperature =
            temp_scale * (self.problem.weights.w3 / TEMP_WEIGHT_DIVISOR).max(1e-9);
        let temperature =
            TemperatureSchedule::new(start, deadline, start_temperature, end_temperature);

        loop {
            let elapsed = self.timer.elapsed_seconds();
            if elapsed >= deadline {
                break;
            }
            state.stats.iter += 1;
            if state.stats.iter % BEST_EXCHANGE_INTERVAL == 0 {
                let shared = shared.lock().unwrap();
                if shared.score + self.problem.weights.w1 + 1e-9 < state.current_score {
                    state.adopt_external(&shared);
                    tabu.insert_schedule(&state.current);
                }
            }

            let temp = temperature.temperature(elapsed);
            let accept_threshold = state.current_score - temp * rng.nextf().ln();
            let neighbor = sample_neighbor(&mut rng, NEIGHBOR_PROBS);
            let neighbor_idx = neighbor.index();
            let neighbor_start = Instant::now();
            state.stats.neighbor_stats[neighbor_idx].selected += 1;

            let candidate = match neighbor {
                NeighborKind::LargeReconstruct => try_large_reconstruct(
                    self.problem,
                    self.pre,
                    &state.current,
                    &mut rng,
                    accept_threshold,
                ),
                NeighborKind::Shift => {
                    try_shift_neighbor(self.problem, self.pre, &state.current, &mut rng, None)
                }
                NeighborKind::Move => {
                    try_move_neighbor(self.problem, self.pre, &state.current, &mut rng, None, None)
                }
                NeighborKind::Rotate => {
                    try_rotate_neighbor(self.problem, self.pre, &state.current, &mut rng, None)
                }
                NeighborKind::Swap => {
                    try_swap_neighbor(self.problem, self.pre, &state.current, &mut rng)
                }
            };
            let Some(candidate) = candidate else {
                state.stats.neighbor_stats[neighbor_idx].time_sec +=
                    neighbor_start.elapsed().as_secs_f64();
                continue;
            };
            state.stats.neighbor_stats[neighbor_idx].succeeded += 1;

            let candidate_key = ScheduleTabu::key(&candidate);
            let is_tabu = tabu.contains(candidate_key);
            let score = score_schedule(self.problem, self.pre, &candidate);
            if is_tabu && score + 1e-9 >= state.best_score {
                state.stats.neighbor_stats[neighbor_idx].time_sec +=
                    neighbor_start.elapsed().as_secs_f64();
                continue;
            }

            let outcome =
                state.accept_candidate(candidate, score, score, accept_threshold, neighbor_idx);
            if outcome.accepted {
                tabu.insert(candidate_key);
            }
            if outcome.new_best {
                log!(
                    self.timer,
                    "worker {} new best score: {:.3}",
                    worker_id,
                    state.best_score
                );
                let mut shared = shared.lock().unwrap();
                if state.best_score + 1e-9 < shared.score {
                    shared.score = state.best_score;
                    shared.blocks = state.best.clone();
                }
            }
            state.stats.neighbor_stats[neighbor_idx].time_sec +=
                neighbor_start.elapsed().as_secs_f64();
        }

        AnnealingResult {
            worker_id,
            state: OptimizeState {
                score: state.best_score,
                blocks: state.best,
            },
            current_score: state.current_score,
            stats: state.stats,
        }
    }
}

fn get_temp_scale(worker_id: usize, worker_count: usize) -> f64 {
    if worker_count <= 1 {
        return 1.0;
    }
    let ratio = worker_id as f64 / (worker_count - 1) as f64;
    1. + WORKER_TEMP_SCALE * ratio.powf(2.)
}
