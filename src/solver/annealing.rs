use std::{
    collections::{HashSet, VecDeque},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use rayon::prelude::*;

use crate::utils::{
    random::{RandPcg64Mcg, Random},
    time::Timer,
};

const EPS: f64 = 1e-9;
const STATUS_LOG_INTERVAL_SECONDS: f64 = 1.0;

pub(crate) trait AnnealingState: Clone + Send + Sync {
    fn annealing_score(&self) -> f64;

    fn tabu_key(&self) -> Option<u64>;
}

pub(crate) struct SharedBest<S: AnnealingState> {
    state: Mutex<S>,
    revision: AtomicU64,
}

impl<S: AnnealingState> SharedBest<S> {
    pub(crate) fn new(state: S) -> Self {
        Self {
            state: Mutex::new(state),
            revision: AtomicU64::new(0),
        }
    }

    pub(crate) fn update(&self, candidate: &S) -> Option<u64> {
        let mut best = self.state.lock().unwrap();
        if candidate.annealing_score() + EPS >= best.annealing_score() {
            return None;
        }
        best.clone_from(candidate);
        Some(self.revision.fetch_add(1, Ordering::Release) + 1)
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision.load(Ordering::Acquire)
    }

    pub(crate) fn score(&self) -> f64 {
        self.state.lock().unwrap().annealing_score()
    }

    pub(crate) fn snapshot(&self) -> (S, u64) {
        let best = self.state.lock().unwrap();
        (best.clone(), self.revision.load(Ordering::Relaxed))
    }

    pub(crate) fn get_if_better(
        &self,
        current_score: f64,
        exchange_threshold: f64,
        last_imported_revision: Option<u64>,
    ) -> Option<(S, u64)> {
        let best = self.state.lock().unwrap();
        let revision = self.revision.load(Ordering::Relaxed);
        (last_imported_revision != Some(revision)
            && best.annealing_score() + exchange_threshold + EPS < current_score)
            .then(|| (best.clone(), revision))
    }

    pub(crate) fn into_inner(self) -> S {
        self.state.into_inner().unwrap()
    }
}

struct AnnealingTemperature {
    start_time: f64,
    deadline: f64,
    base: f64,
    scale: (f64, f64),
    reheat: Option<ReheatParams>,
    reheat_started_iteration: Option<usize>,
}

impl AnnealingTemperature {
    fn new(timer: Timer, deadline: f64, params: &AnnealingParams) -> Self {
        Self {
            start_time: timer.elapsed_seconds(),
            deadline,
            base: params.initial_base,
            scale: params.temperature_scale,
            reheat: params.reheat,
            reheat_started_iteration: None,
        }
    }

    fn normal_progress(&self, elapsed: f64) -> f64 {
        ((elapsed - self.start_time) / (self.deadline - self.start_time).max(1e-4)).clamp(0.0, 1.0)
    }

    fn current(&self, elapsed: f64, iteration: usize) -> f64 {
        let progress = self.normal_progress(elapsed);
        let normal_scale = self.scale.0 * (self.scale.1 / self.scale.0).powf(progress);
        let scale = match (self.reheat, self.reheat_started_iteration) {
            (Some(reheat), Some(start)) => {
                let recovery =
                    iteration.saturating_sub(start) as f64 / reheat.recovery_iterations as f64;
                let recovery = recovery.clamp(0.0, 1.0);
                reheat.temperature_scale * (normal_scale / reheat.temperature_scale).powf(recovery)
            }
            _ => normal_scale,
        };
        self.base * scale
    }

    fn update_base(&mut self, score: f64, block_count: usize) {
        self.base = score / block_count as f64;
    }

    fn exchange_threshold(&self, scale: f64) -> f64 {
        self.base * scale
    }

    fn start_reheat(&mut self, iteration: usize) {
        self.reheat_started_iteration = Some(iteration);
    }

    fn cancel_reheat(&mut self) {
        self.reheat_started_iteration = None;
    }
}

pub(crate) struct AnnealingWorkerContext {
    pub(crate) rng: RandPcg64Mcg,
    deadline: f64,
    iterations: usize,
}

impl AnnealingWorkerContext {
    pub(crate) fn new(deadline: f64, rng_seed: u64) -> Self {
        Self {
            rng: RandPcg64Mcg::new(rng_seed),
            deadline,
            iterations: 0,
        }
    }

    pub(crate) fn next(&mut self, timer: Timer) -> Option<f64> {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= self.deadline {
            return None;
        }
        self.iterations += 1;
        Some(elapsed)
    }

    pub(crate) fn should_exchange(&self, interval: usize) -> bool {
        self.iterations % interval == 0
    }

    pub(crate) fn iterations(&self) -> usize {
        self.iterations
    }
}

pub(crate) fn acceptance_threshold(
    current_score: f64,
    temperature: f64,
    rng: &mut impl Random,
) -> f64 {
    if temperature <= 0.0 {
        return current_score;
    }
    current_score - temperature * rng.next_f64().ln()
}

#[derive(Clone, Copy)]
pub(crate) struct ReheatParams {
    pub(crate) stagnation_iterations: usize,
    pub(crate) recovery_iterations: usize,
    pub(crate) temperature_scale: f64,
}

pub(crate) struct AnnealingParams {
    pub(crate) initial_base: f64,
    pub(crate) block_count: usize,
    pub(crate) temperature_scale: (f64, f64),
    pub(crate) exchange_interval: usize,
    pub(crate) exchange_threshold_scale: f64,
    pub(crate) reheat: Option<ReheatParams>,
    pub(crate) tabu_capacity: usize,
}

pub(crate) struct AnnealingAttempt<S> {
    pub(crate) neighbor_kind: usize,
    pub(crate) candidate: Option<S>,
}

#[derive(Clone, Default)]
pub(crate) struct NeighborStats {
    pub(crate) selected: usize,
    pub(crate) succeeded: usize,
    pub(crate) improved: usize,
    pub(crate) accepted: usize,
    pub(crate) improved_delta_sum: f64,
    pub(crate) time_sec: f64,
}

fn format_neighbor_stats(stats: &[NeighborStats], kinds: &[&str]) -> String {
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

    debug_assert_eq!(stats.len(), kinds.len());
    kinds
        .iter()
        .zip(stats)
        .map(|(kind, stat)| {
            format!(
                "  {kind:<16}: selected={:7}, succeeded={:7} ({:7.3}%), improved={:7} ({:7.3}%, avg={:10.2}), accepted={:7} ({:7.3}%), time={:7.3}s, avg={:7.3}ms",
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

pub(crate) struct WorkerSummary {
    pub(crate) worker_id: usize,
    pub(crate) iterations: usize,
    pub(crate) accepted: usize,
    pub(crate) improved: usize,
    pub(crate) best_imports: usize,
    pub(crate) best_returns: usize,
    pub(crate) reheats: usize,
    pub(crate) base: f64,
    pub(crate) temperature: f64,
    pub(crate) current_score: f64,
    pub(crate) local_best_score: f64,
    pub(crate) neighbor_stats: Vec<NeighborStats>,
}

pub(crate) trait AnnealingDelegate: Sync {
    type State: AnnealingState;
    type Output;

    fn initial_state(&self) -> Self::State;

    fn name(&self) -> &'static str;

    fn neighbor_kinds(&self) -> &'static [&'static str];

    fn propose(
        &self,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State>;

    fn on_shared_best(&self, _state: &Self::State, _timer: Timer) {}

    fn finish(&self, state: Self::State) -> Self::Output;
}

struct TabuList {
    capacity: usize,
    queue: VecDeque<u64>,
    set: HashSet<u64>,
}

impl TabuList {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            queue: VecDeque::with_capacity(capacity),
            set: HashSet::with_capacity(capacity * 2),
        }
    }

    fn contains(&self, key: u64) -> bool {
        self.capacity > 0 && self.set.contains(&key)
    }

    fn insert(&mut self, key: u64) {
        if self.capacity == 0 || !self.set.insert(key) {
            return;
        }
        self.queue.push_back(key);
        if self.queue.len() > self.capacity {
            if let Some(old_key) = self.queue.pop_front() {
                self.set.remove(&old_key);
            }
        }
    }
}

struct WorkerState<S> {
    current: S,
    local_best_score: f64,
    last_imported_revision: Option<u64>,
    last_seen_revision: u64,
    best_exploration_started_iteration: usize,
    tabu: TabuList,
}

fn apply_shared_best<S: AnnealingState>(local: &mut WorkerState<S>, best: S, revision: u64) {
    let score = best.annealing_score();
    if let Some(key) = best.tabu_key() {
        local.tabu.insert(key);
    }
    local.current = best;
    local.local_best_score = local.local_best_score.min(score);
    local.last_imported_revision = Some(revision);
}

pub(crate) struct Annealer<D> {
    pub(crate) deadline: f64,
    pub(crate) worker_count: usize,
    pub(crate) rng_seed: u64,
    pub(crate) params: AnnealingParams,
    pub(crate) delegate: D,
}

impl<D: AnnealingDelegate> Annealer<D> {
    pub(crate) fn new(
        deadline: f64,
        worker_count: usize,
        rng_seed: u64,
        params: AnnealingParams,
        delegate: D,
    ) -> Self {
        Self {
            deadline,
            worker_count,
            rng_seed,
            params,
            delegate,
        }
    }

    pub(crate) fn run(self, timer: Timer) -> D::Output {
        let initial_state = self.delegate.initial_state();
        assert!(self.worker_count > 0);
        assert!(self.params.exchange_interval > 0);

        let shared = SharedBest::new(initial_state.clone());
        let worker_results: Vec<_> = (0..self.worker_count)
            .into_par_iter()
            .map(|worker_id| self.run_worker(&initial_state, &shared, timer, worker_id))
            .collect();
        let state = shared.into_inner();
        for worker in &worker_results {
            log!(
                "[{:.4}] [{:8} worker={}] iter={:8}, accepted={:8}, improved={:8}, best_imports={:5}, best_returns={:5}, reheats={:5}, current={:.3}, local_best={:.3}, base={:.3}, temperature={:.3}\nneighbor stats:\n{}",
                timer.elapsed_seconds(),
                self.delegate.name(),
                worker.worker_id,
                worker.iterations,
                worker.accepted,
                worker.improved,
                worker.best_imports,
                worker.best_returns,
                worker.reheats,
                worker.current_score,
                worker.local_best_score,
                worker.base,
                worker.temperature,
                format_neighbor_stats(&worker.neighbor_stats, self.delegate.neighbor_kinds()),
            );
        }
        log!(
            "[{:.4}] [{}] result: best={:.3}",
            timer.elapsed_seconds(),
            self.delegate.name(),
            state.annealing_score(),
        );
        self.delegate.finish(state)
    }

    fn run_worker(
        &self,
        initial_state: &D::State,
        shared: &SharedBest<D::State>,
        timer: Timer,
        worker_id: usize,
    ) -> WorkerSummary {
        let score = initial_state.annealing_score();
        let mut tabu = TabuList::new(self.params.tabu_capacity);
        if let Some(key) = initial_state.tabu_key() {
            tabu.insert(key);
        }
        let mut local = WorkerState {
            current: initial_state.clone(),
            local_best_score: score,
            last_imported_revision: Some(0),
            last_seen_revision: 0,
            best_exploration_started_iteration: 1,
            tabu,
        };
        let mut temperature = AnnealingTemperature::new(timer, self.deadline, &self.params);
        let mut context = AnnealingWorkerContext::new(
            self.deadline,
            self.rng_seed.wrapping_add(worker_id as u64),
        );
        let mut accepted = 0usize;
        let mut improved = 0usize;
        let mut best_imports = 0usize;
        let mut best_returns = 0usize;
        let mut reheats = 0usize;
        let mut neighbor_stats =
            vec![NeighborStats::default(); self.delegate.neighbor_kinds().len()];
        let mut next_status_time = (timer.elapsed_seconds() / STATUS_LOG_INTERVAL_SECONDS).floor()
            * STATUS_LOG_INTERVAL_SECONDS
            + STATUS_LOG_INTERVAL_SECONDS;

        loop {
            let Some(elapsed) = context.next(timer) else {
                break;
            };
            if let Some(reheat) = self.params.reheat {
                let revision = shared.revision();
                if revision != local.last_seen_revision {
                    local.last_seen_revision = revision;
                    local.best_exploration_started_iteration = context.iterations();
                    temperature.cancel_reheat();
                }

                let return_due = context
                    .iterations()
                    .saturating_sub(local.best_exploration_started_iteration)
                    >= reheat.stagnation_iterations;
                if return_due {
                    let revision = self.import_shared_best(&mut local, shared);
                    local.last_seen_revision = revision;
                    local.best_exploration_started_iteration = context.iterations();
                    temperature
                        .update_base(local.current.annealing_score(), self.params.block_count);
                    best_returns += 1;
                    reheats += 1;
                    temperature.start_reheat(context.iterations());
                    log!(
                        "[{elapsed:.4}] [{} worker={worker_id}] reheat: revision={revision}, iter={}, recovery_iterations={}, temperature_scale={:.6}, base={:.6}",
                        self.delegate.name(),
                        context.iterations(),
                        reheat.recovery_iterations,
                        reheat.temperature_scale,
                        temperature.base,
                    );
                }
            }

            if context.should_exchange(self.params.exchange_interval) {
                temperature.update_base(shared.score(), self.params.block_count);
                let exchange_threshold =
                    temperature.exchange_threshold(self.params.exchange_threshold_scale);
                if self.exchange_shared(&mut local, shared, exchange_threshold) {
                    best_imports += 1;
                }
            }

            let current_score = local.current.annealing_score();
            let current_temperature = temperature.current(elapsed, context.iterations());
            if elapsed >= next_status_time {
                log!(
                    "[{elapsed:.4}] [{} worker={worker_id}] iter={:8}, current={current_score:.3}, base={:.3}, temperature={current_temperature:.3}, reheating={}",
                    self.delegate.name(),
                    context.iterations(),
                    temperature.base,
                    temperature.reheat_started_iteration.is_some_and(|start| {
                        context.iterations().saturating_sub(start)
                            < self
                                .params
                                .reheat
                                .map_or(0, |reheat| reheat.recovery_iterations)
                    }),
                );
                next_status_time = (elapsed / STATUS_LOG_INTERVAL_SECONDS).floor()
                    * STATUS_LOG_INTERVAL_SECONDS
                    + STATUS_LOG_INTERVAL_SECONDS;
            }
            let accept_threshold =
                acceptance_threshold(current_score, current_temperature, &mut context.rng);
            let neighbor_start = Instant::now();
            let attempt = self
                .delegate
                .propose(&local.current, accept_threshold, &mut context.rng);
            debug_assert!(attempt.neighbor_kind < neighbor_stats.len());
            let stats = &mut neighbor_stats[attempt.neighbor_kind];
            stats.selected += 1;

            let Some(candidate) = attempt.candidate else {
                stats.time_sec += neighbor_start.elapsed().as_secs_f64();
                continue;
            };
            stats.succeeded += 1;
            let candidate_score = candidate.annealing_score();
            let candidate_tabu_key = candidate.tabu_key();
            let tabu = candidate_tabu_key.is_some_and(|key| local.tabu.contains(key));
            if tabu && candidate_score + EPS >= local.local_best_score {
                stats.time_sec += neighbor_start.elapsed().as_secs_f64();
                continue;
            }

            let delta = candidate_score - current_score;
            if delta < -EPS {
                stats.improved += 1;
                stats.improved_delta_sum += -delta;
            }
            if candidate_score > accept_threshold {
                stats.time_sec += neighbor_start.elapsed().as_secs_f64();
                continue;
            }

            local.current = candidate;
            accepted += 1;
            stats.accepted += 1;
            if let Some(key) = candidate_tabu_key {
                local.tabu.insert(key);
            }

            if candidate_score + EPS < local.local_best_score {
                local.local_best_score = candidate_score;
                improved += 1;
                log!(
                    "[{:.4}] [{}]  local best: worker={}, iter={:8}, score={:.3}",
                    timer.elapsed_seconds(),
                    self.delegate.name(),
                    worker_id,
                    context.iterations(),
                    candidate_score,
                );
                if let Some(revision) = shared.update(&local.current) {
                    local.last_imported_revision = Some(revision);
                    if self.params.reheat.is_some() {
                        local.last_seen_revision = revision;
                        local.best_exploration_started_iteration =
                            context.iterations().saturating_add(1);
                        temperature.cancel_reheat();
                    }
                    self.delegate.on_shared_best(&local.current, timer);
                    log!(
                        "[{:.4}] [{}] shared best: worker={}, iter={:8}, score={:.3}",
                        timer.elapsed_seconds(),
                        self.delegate.name(),
                        worker_id,
                        context.iterations(),
                        candidate_score,
                    );
                }
            }
            stats.time_sec += neighbor_start.elapsed().as_secs_f64();
        }

        WorkerSummary {
            worker_id,
            iterations: context.iterations(),
            accepted,
            improved,
            best_imports,
            best_returns,
            reheats,
            base: temperature.base,
            temperature: temperature.current(timer.elapsed_seconds(), context.iterations()),
            current_score: local.current.annealing_score(),
            local_best_score: local.local_best_score,
            neighbor_stats,
        }
    }

    fn import_shared_best(
        &self,
        local: &mut WorkerState<D::State>,
        shared: &SharedBest<D::State>,
    ) -> u64 {
        let (best, revision) = shared.snapshot();
        apply_shared_best(local, best, revision);
        revision
    }

    fn exchange_shared(
        &self,
        local: &mut WorkerState<D::State>,
        shared: &SharedBest<D::State>,
        exchange_threshold: f64,
    ) -> bool {
        if let Some((best, revision)) = shared.get_if_better(
            local.current.annealing_score(),
            exchange_threshold,
            local.last_imported_revision,
        ) {
            apply_shared_best(local, best, revision);
            true
        } else {
            false
        }
    }
}
