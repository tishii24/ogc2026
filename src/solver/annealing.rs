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

impl AnnealingStats {
    fn merge(&mut self, other: Self) {
        self.iter += other.iter;
        self.accepted += other.accepted;
        self.improved += other.improved;
        for (stats, other) in self.neighbor_stats.iter_mut().zip(other.neighbor_stats) {
            stats.selected += other.selected;
            stats.succeeded += other.succeeded;
            stats.improved += other.improved;
            stats.accepted += other.accepted;
            stats.improved_delta_sum += other.improved_delta_sum;
            stats.time_sec += other.time_sec;
        }
    }
}

struct ScheduleAnnealingState {
    current: Vec<ScheduledBlock>,
    current_score: f64,
    current_metric: f64,
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
            current_metric: metric,
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
        self.current_metric = metric;
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

    fn adopt_external(&mut self, blocks: &[ScheduledBlock], score: f64, metric: f64) {
        self.current = blocks.to_vec();
        self.current_score = score;
        self.current_metric = metric;
        if metric + 1e-9 < self.best_metric {
            self.best = blocks.to_vec();
            self.best_score = score;
            self.best_metric = metric;
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

struct SharedBayBest {
    blocks: Vec<ScheduledBlock>,
    tardiness: i64,
}

struct BayAnnealingWorkerResult {
    worker_id: usize,
    active_bays: usize,
    stats: AnnealingStats,
}

pub struct BayAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: PrecedenceConstraints,
    target_tardiness: Vec<i64>,
    timer: Timer,
}

impl<'a> BayAnnealing<'a> {
    pub fn new(
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

    pub fn run(&self, initial: OptimizeState, deadline: f64) -> OptimizeState {
        if self.timer.elapsed_seconds() >= deadline {
            return initial;
        }

        let mut initial_bays = vec![Vec::new(); self.problem.bays.len()];
        for block in &initial.blocks {
            initial_bays[block.bay_id].push(*block);
        }
        let active_bay_count = initial_bays
            .iter()
            .enumerate()
            .filter(|(bay_id, blocks)| {
                bay_tardiness(self.problem, blocks) > self.target_tardiness[*bay_id]
            })
            .count();
        if active_bay_count == 0 {
            log!(self.timer, "bay annealing: active_bays=0, workers=0");
            return initial;
        }

        let shared: Vec<_> = initial_bays
            .iter()
            .map(|blocks| {
                Mutex::new(SharedBayBest {
                    tardiness: bay_tardiness(self.problem, blocks),
                    blocks: blocks.clone(),
                })
            })
            .collect();
        let worker_count = rayon::current_num_threads().clamp(1, MAX_WORKER_COUNT);
        log!(
            self.timer,
            "bay annealing: active_bays={}, workers={}",
            active_bay_count,
            worker_count
        );
        let results: Vec<_> = (0..worker_count)
            .into_par_iter()
            .map(|worker_id| {
                self.run_worker(&shared, &initial_bays, deadline, worker_id, worker_count)
            })
            .collect();

        for result in results {
            eprintln!(
                "[{:.4}] [bay-worker={}] iter={}, accepted={}, improved={}, active_bays={}\nneighbor stats:\n{}",
                self.timer.elapsed_seconds(),
                result.worker_id,
                result.stats.iter,
                result.stats.accepted,
                result.stats.improved,
                result.active_bays,
                format_neighbor_stats(&result.stats.neighbor_stats, BAY_NEIGHBOR_PROBS),
            );
        }

        let mut blocks = Vec::with_capacity(self.problem.blocks.len());
        for (bay_id, shared) in shared.into_iter().enumerate() {
            let best = shared.into_inner().unwrap();
            eprintln!(
                "[{:.4}] [bay={}] best_tardiness={}, target={}",
                self.timer.elapsed_seconds(),
                bay_id,
                best.tardiness,
                self.target_tardiness[bay_id],
            );
            blocks.extend(best.blocks);
        }
        let score = score_schedule(self.problem, self.pre, &blocks);
        OptimizeState { score, blocks }
    }

    fn run_worker(
        &self,
        shared: &[Mutex<SharedBayBest>],
        initial_bays: &[Vec<ScheduledBlock>],
        deadline: f64,
        worker_id: usize,
        worker_count: usize,
    ) -> BayAnnealingWorkerResult {
        let mut rng =
            RandPcg64Mcg::new(RNG_SEED.wrapping_add(10_000).wrapping_add(worker_id as u64));
        let mut states: Vec<_> = initial_bays
            .iter()
            .map(|blocks| {
                let tardiness = bay_tardiness(self.problem, blocks);
                ScheduleAnnealingState::new(
                    blocks.clone(),
                    self.problem.weights.w1 * tardiness as f64,
                    tardiness as f64,
                )
            })
            .collect();
        let mut active_bays: Vec<_> = (0..states.len())
            .filter(|&bay_id| states[bay_id].best_metric > self.target_tardiness[bay_id] as f64)
            .collect();
        let temp_scale = get_temp_scale(worker_id, worker_count);
        let start = self.timer.elapsed_seconds();
        let start_temperature =
            temp_scale * (self.problem.weights.w1 / TEMP_WEIGHT_DIVISOR).max(1e-9);
        let temperature = TemperatureSchedule::new(
            start,
            deadline,
            start_temperature,
            (start_temperature * BAY_END_TEMPERATURE_RATIO).max(1e-9),
        );
        let mut turn = 0usize;

        while !active_bays.is_empty() {
            let elapsed = self.timer.elapsed_seconds();
            if elapsed >= deadline {
                break;
            }
            turn += 1;
            if turn % BAY_BEST_EXCHANGE_INTERVAL == 0 {
                self.exchange_shared(shared, &mut states, &mut active_bays);
                if active_bays.is_empty() {
                    break;
                }
            }

            let active_index = rng.gen_index(active_bays.len());
            let bay_id = active_bays[active_index];
            let state = &mut states[bay_id];
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
            let outcome = state.accept_candidate(
                candidate,
                score,
                candidate_tardiness as f64,
                accept_threshold,
                neighbor_idx,
            );
            let mut target_reached = false;
            if outcome.new_best {
                let best_tardiness = state.best_metric as i64;
                let mut shared = shared[bay_id].lock().unwrap();
                if best_tardiness < shared.tardiness {
                    shared.tardiness = best_tardiness;
                    shared.blocks.clone_from(&state.best);
                    log!(
                        self.timer,
                        "worker {} bay {} new shared best tardiness: {} (target={})",
                        worker_id,
                        bay_id,
                        best_tardiness,
                        self.target_tardiness[bay_id]
                    );
                    target_reached = best_tardiness <= self.target_tardiness[bay_id];
                    if target_reached {
                        log!(
                            self.timer,
                            "bay {} target reached: tardiness={}",
                            bay_id,
                            best_tardiness
                        );
                    }
                }
            }
            if target_reached {
                active_bays.swap_remove(active_index);
            }
            state.stats.neighbor_stats[neighbor_idx].time_sec +=
                neighbor_start.elapsed().as_secs_f64();
        }

        let active_bay_count = active_bays.len();
        let mut stats = AnnealingStats::default();
        for state in states {
            stats.merge(state.stats);
        }
        BayAnnealingWorkerResult {
            worker_id,
            active_bays: active_bay_count,
            stats,
        }
    }

    fn exchange_shared(
        &self,
        shared: &[Mutex<SharedBayBest>],
        states: &mut [ScheduleAnnealingState],
        active_bays: &mut Vec<usize>,
    ) {
        active_bays.clear();
        for (bay_id, (shared, state)) in shared.iter().zip(states).enumerate() {
            let shared = shared.lock().unwrap();
            if shared.tardiness as f64 + 1e-9 < state.current_metric {
                state.adopt_external(
                    &shared.blocks,
                    self.problem.weights.w1 * shared.tardiness as f64,
                    shared.tardiness as f64,
                );
            }
            if shared.tardiness > self.target_tardiness[bay_id] {
                active_bays.push(bay_id);
            }
        }
    }
}

pub struct GlobalAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    timer: Timer,
}

impl<'a> GlobalAnnealing<'a> {
    pub fn new(problem: &'a Problem, pre: &'a Precompute, timer: Timer) -> Self {
        Self {
            problem,
            pre,
            timer,
        }
    }

    pub fn run(&self, initial: OptimizeState, deadline: f64) -> OptimizeState {
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
                    state.adopt_external(&shared.blocks, shared.score, shared.score);
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
