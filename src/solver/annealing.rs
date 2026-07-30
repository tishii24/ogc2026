use std::{
    collections::{HashSet, VecDeque},
    sync::Mutex,
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

    fn has_tardiness(&self) -> bool;

    fn tabu_key(&self) -> Option<u64>;
}

struct SharedBestInner<S> {
    state: S,
    revision: u64,
}

pub(crate) struct SharedBest<S: AnnealingState> {
    inner: Mutex<SharedBestInner<S>>,
}

impl<S: AnnealingState> SharedBest<S> {
    pub(crate) fn new(state: S) -> Self {
        Self {
            inner: Mutex::new(SharedBestInner { state, revision: 0 }),
        }
    }

    pub(crate) fn update(&self, candidate: &S) -> Option<u64> {
        let mut best = self.inner.lock().unwrap();
        if candidate.annealing_score() + EPS >= best.state.annealing_score() {
            return None;
        }
        best.state.clone_from(candidate);
        best.revision += 1;
        Some(best.revision)
    }

    pub(crate) fn get_if_better(
        &self,
        current_score: f64,
        exchange_threshold: f64,
        last_imported_revision: Option<u64>,
    ) -> Option<(S, u64)> {
        let best = self.inner.lock().unwrap();
        (last_imported_revision != Some(best.revision)
            && best.state.annealing_score() + exchange_threshold + EPS < current_score)
            .then(|| (best.state.clone(), best.revision))
    }

    pub(crate) fn into_inner(self) -> S {
        self.inner.into_inner().unwrap().state
    }
}

pub(crate) struct TemperatureSchedule {
    start_time: f64,
    deadline: f64,
}

impl TemperatureSchedule {
    pub(crate) fn new(start_time: f64, deadline: f64) -> Self {
        Self {
            start_time,
            deadline,
        }
    }

    pub(crate) fn progress(&self, elapsed: f64) -> f64 {
        ((elapsed - self.start_time) / (self.deadline - self.start_time).max(1e-4)).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TemperatureScheduleKind {
    Linear,
    Cosine,
    Geometric,
}

fn temperature(schedule: TemperatureScheduleKind, range: (f64, f64), progress: f64) -> f64 {
    match schedule {
        TemperatureScheduleKind::Linear => range.0 + (range.1 - range.0) * progress,
        TemperatureScheduleKind::Cosine => {
            let ratio = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            range.1 + (range.0 - range.1) * ratio
        }
        TemperatureScheduleKind::Geometric => range.0 * (range.1 / range.0).powf(progress),
    }
}

pub(crate) struct AnnealingWorkerContext {
    pub(crate) rng: RandPcg64Mcg,
    temperature: TemperatureSchedule,
    deadline: f64,
    iterations: usize,
}

impl AnnealingWorkerContext {
    pub(crate) fn new(timer: Timer, deadline: f64, rng_seed: u64) -> Self {
        Self {
            rng: RandPcg64Mcg::new(rng_seed),
            temperature: TemperatureSchedule::new(timer.elapsed_seconds(), deadline),
            deadline,
            iterations: 0,
        }
    }

    pub(crate) fn next(&mut self, timer: Timer) -> Option<(f64, f64)> {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= self.deadline {
            return None;
        }
        self.iterations += 1;
        Some((elapsed, self.temperature.progress(elapsed)))
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
    current_score - temperature * rng.next_f64().ln()
}

pub(crate) fn worker_temperature_scale(worker_id: usize, worker_count: usize, scale: f64) -> f64 {
    if worker_count <= 1 {
        return 1.0;
    }
    let ratio = worker_id as f64 / (worker_count - 1) as f64;
    1.0 + scale * ratio.powf(2.0)
}

pub(crate) struct AnnealingRegimeParams {
    pub(crate) temperature_schedule: TemperatureScheduleKind,
    pub(crate) temperature: (f64, f64),
    pub(crate) exchange_threshold: f64,
}

pub(crate) struct AnnealingParams {
    pub(crate) exchange_interval: usize,
    pub(crate) positive_tardiness: AnnealingRegimeParams,
    pub(crate) zero_tardiness: AnnealingRegimeParams,
    pub(crate) worker_temperature_scale: f64,
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
    pub(crate) positive_tardiness_temperature: (f64, f64),
    pub(crate) zero_tardiness_temperature: (f64, f64),
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
    tabu: TabuList,
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
                "[{:.4}] [{:8} worker={}] iter={:8}, accepted={:8}, improved={:8}, current={:.3}, local_best={:.3}, temp(z1>0)={:.6}->{:.6}, temp(z1=0)={:.6}->{:.6}\nneighbor stats:\n{}",
                timer.elapsed_seconds(),
                self.delegate.name(),
                worker.worker_id,
                worker.iterations,
                worker.accepted,
                worker.improved,
                worker.current_score,
                worker.local_best_score,
                worker.positive_tardiness_temperature.0,
                worker.positive_tardiness_temperature.1,
                worker.zero_tardiness_temperature.0,
                worker.zero_tardiness_temperature.1,
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
            last_imported_revision: None,
            tabu,
        };
        let scale = worker_temperature_scale(
            worker_id,
            self.worker_count,
            self.params.worker_temperature_scale,
        );
        let positive_tardiness_temperature = (
            self.params.positive_tardiness.temperature.0 * scale,
            self.params.positive_tardiness.temperature.1 * scale,
        );
        let zero_tardiness_temperature = (
            self.params.zero_tardiness.temperature.0 * scale,
            self.params.zero_tardiness.temperature.1 * scale,
        );
        let mut context = AnnealingWorkerContext::new(
            timer,
            self.deadline,
            self.rng_seed.wrapping_add(worker_id as u64),
        );
        let mut accepted = 0usize;
        let mut improved = 0usize;
        let mut neighbor_stats =
            vec![NeighborStats::default(); self.delegate.neighbor_kinds().len()];
        let mut next_status_time = (timer.elapsed_seconds() / STATUS_LOG_INTERVAL_SECONDS).floor()
            * STATUS_LOG_INTERVAL_SECONDS
            + STATUS_LOG_INTERVAL_SECONDS;

        loop {
            let Some((elapsed, progress)) = context.next(timer) else {
                break;
            };
            if context.should_exchange(self.params.exchange_interval) {
                self.exchange_shared(&mut local, shared);
            }

            let current_score = local.current.annealing_score();
            let has_tardiness = local.current.has_tardiness();
            let (regime, temperature_range) = if has_tardiness {
                (
                    &self.params.positive_tardiness,
                    positive_tardiness_temperature,
                )
            } else {
                (&self.params.zero_tardiness, zero_tardiness_temperature)
            };
            let current_temperature =
                temperature(regime.temperature_schedule, temperature_range, progress);
            if elapsed >= next_status_time {
                log!(
                    "[{elapsed:.4}] [{} worker={worker_id}] iter={:8}, current={current_score:.3}, temperature={current_temperature:.6}, regime={}",
                    self.delegate.name(),
                    context.iterations(),
                    if has_tardiness { "T>0" } else { "T=0" },
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
                if shared.update(&local.current).is_some() {
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
            positive_tardiness_temperature,
            zero_tardiness_temperature,
            current_score: local.current.annealing_score(),
            local_best_score: local.local_best_score,
            neighbor_stats,
        }
    }

    fn exchange_shared(&self, local: &mut WorkerState<D::State>, shared: &SharedBest<D::State>) {
        let exchange_threshold = if local.current.has_tardiness() {
            self.params.positive_tardiness.exchange_threshold
        } else {
            self.params.zero_tardiness.exchange_threshold
        };
        if let Some((best, revision)) = shared.get_if_better(
            local.current.annealing_score(),
            exchange_threshold,
            local.last_imported_revision,
        ) {
            let score = best.annealing_score();
            if let Some(key) = best.tabu_key() {
                local.tabu.insert(key);
            }
            local.current = best;
            local.local_best_score = local.local_best_score.min(score);
            local.last_imported_revision = Some(revision);
        }
    }
}
