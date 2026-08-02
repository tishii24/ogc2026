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

struct DeltaWindow {
    capacity: usize,
    quantile: f64,
    values: VecDeque<f64>,
    scratch: Vec<f64>,
    quantile_value: Option<f64>,
}

impl DeltaWindow {
    fn new(capacity: usize, quantile: f64) -> Self {
        Self {
            capacity,
            quantile,
            values: VecDeque::with_capacity(capacity),
            scratch: Vec::with_capacity(capacity),
            quantile_value: None,
        }
    }

    fn push(&mut self, delta: f64) {
        if self.values.len() == self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(delta);
        self.scratch.clear();
        self.scratch.extend(self.values.iter().copied());
        let index = ((self.scratch.len() - 1) as f64 * self.quantile).round() as usize;
        self.scratch
            .select_nth_unstable_by(index, |a, b| a.total_cmp(b));
        self.quantile_value = Some(self.scratch[index]);
    }
}

struct AnnealingTemperature {
    start_time: f64,
    deadline: f64,
    worsening_acceptance: (f64, f64),
    deltas: DeltaWindow,
    reheat: Option<ReheatParams>,
    reheat_started_at: Option<f64>,
    last_improvement_at: f64,
    last_temperature: f64,
    last_target_acceptance: f64,
}

impl AnnealingTemperature {
    fn new(timer: Timer, deadline: f64, params: &AnnealingParams) -> Self {
        let start_time = timer.elapsed_seconds();
        Self {
            start_time,
            deadline,
            worsening_acceptance: params.worsening_acceptance,
            deltas: DeltaWindow::new(params.sample_window, params.worsening_delta_quantile),
            reheat: params.reheat,
            reheat_started_at: None,
            last_improvement_at: start_time,
            last_temperature: 0.0,
            last_target_acceptance: params.worsening_acceptance.0,
        }
    }

    fn progress(&self, elapsed: f64) -> f64 {
        ((elapsed - self.start_time) / (self.deadline - self.start_time).max(1e-4)).clamp(0.0, 1.0)
    }

    fn normal_acceptance(&self, elapsed: f64) -> f64 {
        let progress = self.progress(elapsed);
        let (start, end) = self.worsening_acceptance;
        start * (end / start).powf(progress)
    }

    fn reheat_duration(&self) -> f64 {
        let ratio = self
            .reheat
            .map(|reheat| reheat.stagnation_time_ratio)
            .unwrap_or(0.0);
        ((self.deadline - self.start_time) * ratio).max(1e-4)
    }

    fn current(&mut self, elapsed: f64) -> f64 {
        let normal_acceptance = self.normal_acceptance(elapsed);
        let target_acceptance = match (self.reheat, self.reheat_started_at) {
            (Some(reheat), Some(started_at)) => {
                let progress = (elapsed - started_at) / self.reheat_duration();
                if progress >= 1.0 {
                    self.reheat_started_at = None;
                    self.last_improvement_at = elapsed;
                    normal_acceptance
                } else {
                    reheat.worsening_acceptance
                        * (normal_acceptance / reheat.worsening_acceptance)
                            .powf(progress.clamp(0.0, 1.0))
                }
            }
            _ => normal_acceptance,
        };
        let temperature = self
            .deltas
            .quantile_value
            .map(|delta| -delta / target_acceptance.ln())
            .unwrap_or(0.0);
        self.last_temperature = temperature;
        self.last_target_acceptance = target_acceptance;
        temperature
    }

    fn observe(&mut self, delta: f64) {
        self.deltas.push(delta);
    }

    fn should_reheat(&self, elapsed: f64) -> bool {
        self.reheat.is_some()
            && self.reheat_started_at.is_none()
            && elapsed - self.last_improvement_at >= self.reheat_duration()
    }

    fn start_reheat(&mut self, elapsed: f64) {
        self.reheat_started_at = Some(elapsed);
        self.last_improvement_at = elapsed;
    }

    fn mark_improvement(&mut self, elapsed: f64) {
        self.reheat_started_at = None;
        self.last_improvement_at = elapsed;
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
    pub(crate) stagnation_time_ratio: f64,
    pub(crate) worsening_acceptance: f64,
}

pub(crate) struct AnnealingParams {
    pub(crate) sample_window: usize,
    pub(crate) worsening_delta_quantile: f64,
    pub(crate) worsening_acceptance: (f64, f64),
    pub(crate) exchange_interval: usize,
    pub(crate) exchange_threshold: f64,
    pub(crate) reheat: Option<ReheatParams>,
    pub(crate) tabu_capacity: usize,
}

pub(crate) struct AnnealingAttempt<S> {
    pub(crate) neighbor_kind: usize,
    pub(crate) candidate: Option<S>,
    pub(crate) temperature_sample: bool,
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
                "[{:.4}] [{:8} worker={}] iter={:8}, accepted={:8}, improved={:8}, best_imports={:5}, best_returns={:5}, reheats={:5}, current={:.3}, local_best={:.3}, temperature={:.6}, neighbor stats:\n{}",
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
            last_imported_revision: self.params.reheat.map(|_| 0),
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
            if self.params.reheat.is_some() {
                if local.last_imported_revision != Some(shared.revision()) {
                    self.import_shared_best(&mut local, shared);
                    best_imports += 1;
                    temperature.mark_improvement(elapsed);
                } else if temperature.should_reheat(elapsed) {
                    let revision = self.import_shared_best(&mut local, shared);
                    best_returns += 1;
                    reheats += 1;
                    temperature.start_reheat(elapsed);
                    log!(
                        "[{elapsed:.4}] [{} worker={worker_id}] reheat: revision={revision}, iter={}, target_acceptance={:.4}",
                        self.delegate.name(),
                        context.iterations(),
                        self.params.reheat.unwrap().worsening_acceptance,
                    );
                }
            } else if context.should_exchange(self.params.exchange_interval)
                && self.exchange_shared(&mut local, shared)
            {
                best_imports += 1;
            }

            let current_score = local.current.annealing_score();
            let current_temperature = temperature.current(elapsed);
            if elapsed >= next_status_time {
                log!(
                    "[{elapsed:.4}] [{} worker={worker_id}] iter={:8}, current={current_score:.3}, temperature={current_temperature:.6}, reheating={}",
                    self.delegate.name(),
                    context.iterations(),
                    temperature.reheat_started_at.is_some(),
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
            let delta = candidate_score - current_score;
            if !tabu && attempt.temperature_sample && delta > EPS {
                temperature.observe(delta);
            }
            if tabu && candidate_score + EPS >= local.local_best_score {
                stats.time_sec += neighbor_start.elapsed().as_secs_f64();
                continue;
            }

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
                temperature.mark_improvement(elapsed);
                log!(
                    "[{:.4}] [{}]  local best: worker={}, iter={:8}, score={:.3}",
                    timer.elapsed_seconds(),
                    self.delegate.name(),
                    worker_id,
                    context.iterations(),
                    candidate_score,
                );
                if let Some(revision) = shared.update(&local.current) {
                    if self.params.reheat.is_some() {
                        local.last_imported_revision = Some(revision);
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

        let final_temperature = temperature.current(timer.elapsed_seconds());
        WorkerSummary {
            worker_id,
            iterations: context.iterations(),
            accepted,
            improved,
            best_imports,
            best_returns,
            reheats,
            temperature: final_temperature,
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
    ) -> bool {
        if let Some((best, revision)) = shared.get_if_better(
            local.current.annealing_score(),
            self.params.exchange_threshold,
            local.last_imported_revision,
        ) {
            apply_shared_best(local, best, revision);
            true
        } else {
            false
        }
    }
}
