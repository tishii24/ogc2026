use std::{
    collections::{HashSet, VecDeque},
    sync::Mutex,
    time::Instant,
};

use rayon::prelude::*;

use crate::{
    log,
    utils::base::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
};

#[cfg(feature = "trace-annealing")]
use crate::utils::tracing::{
    AnnealingTraceDiff, AnnealingTraceEvent, AnnealingTraceState, AnnealingTraceWriter,
};

const EPS: f64 = 1e-9;

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

    pub(crate) fn next(&mut self, timer: Timer) -> Option<f64> {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= self.deadline {
            return None;
        }
        self.iterations += 1;
        Some(self.temperature.progress(elapsed))
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
    current_score - temperature * rng.nextf().ln()
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
    pub(crate) reheat_local_best_score_per_block_scale: Option<f64>,
    pub(crate) exchange_threshold: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct ReheatParams {
    pub(crate) stagnation_iterations: usize,
    pub(crate) duration_iterations: usize,
}

pub(crate) struct AnnealingParams {
    pub(crate) exchange_interval: usize,
    pub(crate) positive_tardiness: AnnealingRegimeParams,
    pub(crate) zero_tardiness: AnnealingRegimeParams,
    pub(crate) worker_temperature_scale: f64,
    pub(crate) tabu_capacity: usize,
    pub(crate) reheat: Option<ReheatParams>,
    pub(crate) block_count: usize,
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
    pub(crate) reheats: usize,
    pub(crate) active_domains: usize,
    pub(crate) positive_tardiness_temperature: (f64, f64),
    pub(crate) zero_tardiness_temperature: (f64, f64),
    pub(crate) current_scores: Vec<f64>,
    pub(crate) local_best_scores: Vec<f64>,
    pub(crate) neighbor_stats: Vec<NeighborStats>,
}

pub(crate) trait AnnealingDelegate: Sync {
    type State: AnnealingState;
    type Output;

    fn initial_states(&self) -> Vec<Self::State>;

    fn name(&self) -> &'static str;

    fn neighbor_kinds(&self) -> &'static [&'static str];

    #[cfg(feature = "trace-annealing")]
    fn trace_state(&self, _state: &Self::State) -> Option<AnnealingTraceState> {
        None
    }

    #[cfg(feature = "trace-annealing")]
    fn trace_diff(&self, _before: &Self::State, _after: &Self::State) -> AnnealingTraceDiff {
        AnnealingTraceDiff::default()
    }

    fn propose(
        &self,
        domain: usize,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State>;

    fn is_finished(&self, domain: usize, state: &Self::State) -> bool;

    fn on_shared_best(&self, _domain: usize, _state: &Self::State, _timer: Timer) {}

    fn finish(&self, states: Vec<Self::State>) -> Self::Output;
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

#[derive(Clone, Copy)]
struct ReheatState {
    start_iteration: usize,
    peak_temperature: f64,
}

struct WorkerDomain<S> {
    current: S,
    local_best_score: f64,
    last_progress_iteration: usize,
    reheat: Option<ReheatState>,
    reheat_count: usize,
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
        let initial_states = self.delegate.initial_states();
        assert!(!initial_states.is_empty());
        assert!(self.worker_count > 0);
        assert!(self.params.exchange_interval > 0);

        let shared: Vec<_> = initial_states
            .iter()
            .cloned()
            .map(SharedBest::new)
            .collect();
        let worker_results: Vec<_> = (0..self.worker_count)
            .into_par_iter()
            .map(|worker_id| self.run_worker(&initial_states, &shared, timer, worker_id))
            .collect();
        let states: Vec<_> = shared.into_iter().map(SharedBest::into_inner).collect();
        for worker in &worker_results {
            log!(
                "[{:.4}] [{:8} worker={}] iter={:8}, accepted={:8}, improved={:8}, reheats={:4}, active_domains={:8}, current={:?}, local_best={:?}, temp(z1>0)={:.6}->{:.6}, temp(z1=0)={:.6}->{:.6}\nneighbor stats:\n{}",
                timer.elapsed_seconds(),
                self.delegate.name(),
                worker.worker_id,
                worker.iterations,
                worker.accepted,
                worker.improved,
                worker.reheats,
                worker.active_domains,
                worker.current_scores,
                worker.local_best_scores,
                worker.positive_tardiness_temperature.0,
                worker.positive_tardiness_temperature.1,
                worker.zero_tardiness_temperature.0,
                worker.zero_tardiness_temperature.1,
                format_neighbor_stats(&worker.neighbor_stats, self.delegate.neighbor_kinds()),
            );
        }
        for (domain, state) in states.iter().enumerate() {
            log!(
                "[{:.4}] [{}] result: domain={}, best={:.3}, finished={}",
                timer.elapsed_seconds(),
                self.delegate.name(),
                domain,
                state.annealing_score(),
                self.delegate.is_finished(domain, state),
            );
        }
        self.delegate.finish(states)
    }

    fn run_worker(
        &self,
        initial_states: &[D::State],
        shared: &[SharedBest<D::State>],
        timer: Timer,
        worker_id: usize,
    ) -> WorkerSummary {
        let mut domains: Vec<_> = initial_states
            .iter()
            .cloned()
            .map(|state| {
                let score = state.annealing_score();
                let mut tabu = TabuList::new(self.params.tabu_capacity);
                if let Some(key) = state.tabu_key() {
                    tabu.insert(key);
                }
                WorkerDomain {
                    current: state,
                    local_best_score: score,
                    last_progress_iteration: 0,
                    reheat: None,
                    reheat_count: 0,
                    last_imported_revision: None,
                    tabu,
                }
            })
            .collect();
        let mut active_domains: Vec<_> = initial_states
            .iter()
            .enumerate()
            .filter_map(|(domain, state)| {
                (!self.delegate.is_finished(domain, state)).then_some(domain)
            })
            .collect();

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
        #[cfg(feature = "trace-annealing")]
        let mut trace_writer = initial_states
            .iter()
            .any(|state| self.delegate.trace_state(state).is_some())
            .then(|| AnnealingTraceWriter::new(self.delegate.name(), worker_id))
            .flatten();
        #[cfg(feature = "trace-annealing")]
        if let Some(writer) = &mut trace_writer {
            for (domain, local) in domains.iter().enumerate() {
                if let Some(state) = self.delegate.trace_state(&local.current) {
                    writer.write(
                        timer.elapsed_seconds(),
                        0,
                        domain,
                        AnnealingTraceEvent {
                            event: "start",
                            neighbor: None,
                            temperature: None,
                            accept_threshold: None,
                            before: state,
                            after: state,
                            diff: AnnealingTraceDiff::default(),
                            shared_revision: Some(0),
                        },
                    );
                }
            }
        }

        while !active_domains.is_empty() {
            let Some(progress) = context.next(timer) else {
                break;
            };
            if context.should_exchange(self.params.exchange_interval) {
                self.exchange_shared(
                    &mut domains,
                    shared,
                    &mut active_domains,
                    context.iterations(),
                    |_domain, _before, _after, _revision| {
                        #[cfg(feature = "trace-annealing")]
                        if let Some(writer) = &mut trace_writer
                            && let (Some(before), Some(after)) = (
                                self.delegate.trace_state(_before),
                                self.delegate.trace_state(_after),
                            )
                        {
                            let (temperature_schedule, temperature_range) =
                                if _before.has_tardiness() {
                                    (
                                        self.params.positive_tardiness.temperature_schedule,
                                        positive_tardiness_temperature,
                                    )
                                } else {
                                    (
                                        self.params.zero_tardiness.temperature_schedule,
                                        zero_tardiness_temperature,
                                    )
                                };
                            writer.write(
                                timer.elapsed_seconds(),
                                context.iterations(),
                                _domain,
                                AnnealingTraceEvent {
                                    event: "exchange",
                                    neighbor: None,
                                    temperature: Some(temperature(
                                        temperature_schedule,
                                        temperature_range,
                                        progress,
                                    )),
                                    accept_threshold: None,
                                    before,
                                    after,
                                    diff: self.delegate.trace_diff(_before, _after),
                                    shared_revision: Some(_revision),
                                },
                            );
                        }
                    },
                );
                if active_domains.is_empty() {
                    break;
                }
            }

            let active_index = context.rng.gen_index(active_domains.len());
            let domain = active_domains[active_index];
            let local = &mut domains[domain];
            let current_score = local.current.annealing_score();
            let (regime, temperature_range) = if local.current.has_tardiness() {
                (
                    &self.params.positive_tardiness,
                    positive_tardiness_temperature,
                )
            } else {
                (&self.params.zero_tardiness, zero_tardiness_temperature)
            };
            let base_temperature =
                temperature(regime.temperature_schedule, temperature_range, progress);
            let reheat_scale = regime.reheat_local_best_score_per_block_scale;
            if reheat_scale.is_none() {
                local.reheat = None;
            }
            if let (Some(reheat_params), Some(reheat_scale)) = (self.params.reheat, reheat_scale)
                && local.reheat.is_none()
                && context.iterations() - local.last_progress_iteration
                    >= reheat_params.stagnation_iterations
            {
                let peak_temperature = (local.local_best_score / self.params.block_count as f64
                    * reheat_scale
                    * scale)
                    .max(base_temperature);
                local.reheat = Some(ReheatState {
                    start_iteration: context.iterations(),
                    peak_temperature,
                });
                local.last_progress_iteration = context.iterations();
                local.reheat_count += 1;
                log!(
                    "[{:.4}] [{}] reheat: worker={}, domain={}, iter={}, base={:.3}, peak={:.3}, local_best={:.3}",
                    timer.elapsed_seconds(),
                    self.delegate.name(),
                    worker_id,
                    domain,
                    context.iterations(),
                    base_temperature,
                    peak_temperature,
                    local.local_best_score,
                );
                #[cfg(feature = "trace-annealing")]
                if let Some(writer) = &mut trace_writer
                    && let Some(state) = self.delegate.trace_state(&local.current)
                {
                    writer.write(
                        timer.elapsed_seconds(),
                        context.iterations(),
                        domain,
                        AnnealingTraceEvent {
                            event: "temperature_reheat",
                            neighbor: None,
                            temperature: Some(peak_temperature),
                            accept_threshold: None,
                            before: state,
                            after: state,
                            diff: AnnealingTraceDiff::default(),
                            shared_revision: None,
                        },
                    );
                }
            }
            let current_temperature = if let (Some(reheat_params), Some(reheat)) =
                (self.params.reheat, local.reheat)
            {
                let elapsed = context.iterations() - reheat.start_iteration;
                if elapsed < reheat_params.duration_iterations {
                    let reheat_progress = elapsed as f64 / reheat_params.duration_iterations as f64;
                    let ratio = 0.5 * (1.0 + (std::f64::consts::PI * reheat_progress).cos());
                    base_temperature + (reheat.peak_temperature - base_temperature).max(0.0) * ratio
                } else {
                    local.reheat = None;
                    base_temperature
                }
            } else {
                base_temperature
            };
            let accept_threshold =
                acceptance_threshold(current_score, current_temperature, &mut context.rng);
            let neighbor_start = Instant::now();
            let attempt =
                self.delegate
                    .propose(domain, &local.current, accept_threshold, &mut context.rng);
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

            #[cfg(feature = "trace-annealing")]
            let trace_transition = (
                self.delegate.trace_state(&local.current),
                self.delegate.trace_state(&candidate),
                self.delegate.trace_diff(&local.current, &candidate),
            );
            local.current = candidate;
            accepted += 1;
            stats.accepted += 1;
            if let Some(key) = candidate_tabu_key {
                local.tabu.insert(key);
            }
            #[cfg(feature = "trace-annealing")]
            if let Some(writer) = &mut trace_writer
                && let (Some(before), Some(after), diff) = trace_transition
            {
                writer.write(
                    timer.elapsed_seconds(),
                    context.iterations(),
                    domain,
                    AnnealingTraceEvent {
                        event: "accepted",
                        neighbor: Some(self.delegate.neighbor_kinds()[attempt.neighbor_kind]),
                        temperature: Some(current_temperature),
                        accept_threshold: Some(accept_threshold),
                        before,
                        after,
                        diff,
                        shared_revision: None,
                    },
                );
            }

            if candidate_score + EPS < local.local_best_score {
                local.local_best_score = candidate_score;
                local.last_progress_iteration = context.iterations();
                improved += 1;
                log!(
                    "[{:.4}] [{}]  local best: worker={}, domain={}, iter={:8}, score={:.3}",
                    timer.elapsed_seconds(),
                    self.delegate.name(),
                    worker_id,
                    domain,
                    context.iterations(),
                    candidate_score,
                );
                #[cfg(feature = "trace-annealing")]
                if let Some(writer) = &mut trace_writer
                    && let Some(state) = self.delegate.trace_state(&local.current)
                {
                    writer.write(
                        timer.elapsed_seconds(),
                        context.iterations(),
                        domain,
                        AnnealingTraceEvent {
                            event: "local_best",
                            neighbor: Some(self.delegate.neighbor_kinds()[attempt.neighbor_kind]),
                            temperature: Some(current_temperature),
                            accept_threshold: Some(accept_threshold),
                            before: state,
                            after: state,
                            diff: AnnealingTraceDiff::default(),
                            shared_revision: None,
                        },
                    );
                }
                if let Some(_revision) = shared[domain].update(&local.current) {
                    self.delegate.on_shared_best(domain, &local.current, timer);
                    log!(
                        "[{:.4}] [{}] shared best: worker={}, domain={}, iter={:8}, score={:.3}",
                        timer.elapsed_seconds(),
                        self.delegate.name(),
                        worker_id,
                        domain,
                        context.iterations(),
                        candidate_score,
                    );
                    #[cfg(feature = "trace-annealing")]
                    if let Some(writer) = &mut trace_writer
                        && let Some(state) = self.delegate.trace_state(&local.current)
                    {
                        writer.write(
                            timer.elapsed_seconds(),
                            context.iterations(),
                            domain,
                            AnnealingTraceEvent {
                                event: "shared_best",
                                neighbor: Some(
                                    self.delegate.neighbor_kinds()[attempt.neighbor_kind],
                                ),
                                temperature: Some(current_temperature),
                                accept_threshold: Some(accept_threshold),
                                before: state,
                                after: state,
                                diff: AnnealingTraceDiff::default(),
                                shared_revision: Some(_revision),
                            },
                        );
                    }
                }
                if self.delegate.is_finished(domain, &local.current) {
                    active_domains.swap_remove(active_index);
                }
            }
            stats.time_sec += neighbor_start.elapsed().as_secs_f64();
        }

        #[cfg(feature = "trace-annealing")]
        if let Some(writer) = &mut trace_writer {
            for (domain, local) in domains.iter().enumerate() {
                if let Some(state) = self.delegate.trace_state(&local.current) {
                    writer.write(
                        timer.elapsed_seconds(),
                        context.iterations(),
                        domain,
                        AnnealingTraceEvent {
                            event: "finish",
                            neighbor: None,
                            temperature: None,
                            accept_threshold: None,
                            before: state,
                            after: state,
                            diff: AnnealingTraceDiff::default(),
                            shared_revision: local.last_imported_revision,
                        },
                    );
                }
            }
        }

        WorkerSummary {
            worker_id,
            iterations: context.iterations(),
            accepted,
            improved,
            reheats: domains.iter().map(|domain| domain.reheat_count).sum(),
            active_domains: active_domains.len(),
            positive_tardiness_temperature,
            zero_tardiness_temperature,
            current_scores: domains
                .iter()
                .map(|domain| domain.current.annealing_score())
                .collect(),
            local_best_scores: domains
                .iter()
                .map(|domain| domain.local_best_score)
                .collect(),
            neighbor_stats,
        }
    }

    fn exchange_shared(
        &self,
        domains: &mut [WorkerDomain<D::State>],
        shared: &[SharedBest<D::State>],
        active_domains: &mut Vec<usize>,
        iteration: usize,
        mut on_exchange: impl FnMut(usize, &D::State, &D::State, u64),
    ) {
        active_domains.clear();
        for domain in 0..domains.len() {
            let local = &mut domains[domain];
            let exchange_threshold = if local.current.has_tardiness() {
                self.params.positive_tardiness.exchange_threshold
            } else {
                self.params.zero_tardiness.exchange_threshold
            };
            if let Some((best, revision)) = shared[domain].get_if_better(
                local.current.annealing_score(),
                exchange_threshold,
                local.last_imported_revision,
            ) {
                let score = best.annealing_score();
                on_exchange(domain, &local.current, &best, revision);
                if let Some(key) = best.tabu_key() {
                    local.tabu.insert(key);
                }
                local.current = best;
                local.local_best_score = local.local_best_score.min(score);
                local.last_progress_iteration = iteration;
                local.reheat = None;
                local.last_imported_revision = Some(revision);
            }
            if !self.delegate.is_finished(domain, &local.current) {
                active_domains.push(domain);
            }
        }
    }
}
