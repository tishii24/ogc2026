use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

use rayon::prelude::*;

use crate::{
    EPS,
    utils::{
        random::{RandPcg64Mcg, Random},
        time::Timer,
    },
};
#[cfg(feature = "anneal-visualizer")]
use crate::{
    Problem, ScheduledBlock,
    utils::anneal_visualizer::{AnnealVisualizer, SnapshotMeta},
};

const STATUS_LOG_INTERVAL_SECONDS: f64 = 1.0;

pub(crate) trait AnnealingState: Clone + Send + Sync {
    fn annealing_score(&self) -> f64;

    fn has_tardiness(&self) -> bool;
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

#[derive(Clone, Copy)]
struct TemperatureRegime {
    range: (f64, f64),
}

impl TemperatureRegime {
    fn current(&self, progress: f64) -> f64 {
        self.range.0 * (self.range.1 / self.range.0.max(1e-9)).powf(progress)
    }
}

struct AnnealingTemperature {
    start_time: f64,
    deadline: f64,
    positive_tardiness: TemperatureRegime,
    zero_tardiness: TemperatureRegime,
}

impl AnnealingTemperature {
    fn new(timer: Timer, deadline: f64, params: &AnnealingParams) -> Self {
        Self {
            start_time: timer.elapsed_seconds(),
            deadline,
            positive_tardiness: TemperatureRegime {
                range: (
                    params.positive_tardiness.temperature.0,
                    params.positive_tardiness.temperature.1,
                ),
            },
            zero_tardiness: TemperatureRegime {
                range: (
                    params.zero_tardiness.temperature.0,
                    params.zero_tardiness.temperature.1,
                ),
            },
        }
    }

    fn current(&self, elapsed: f64, has_tardiness: bool) -> f64 {
        let progress = ((elapsed - self.start_time) / (self.deadline - self.start_time).max(1e-4))
            .clamp(0.0, 1.0);
        let regime = if has_tardiness {
            self.positive_tardiness
        } else {
            self.zero_tardiness
        };
        regime.current(progress)
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
    current_score - temperature * rng.next_f64().ln()
}

pub(crate) struct AnnealingRegimeParams {
    pub(crate) temperature: (f64, f64),
    pub(crate) exchange_threshold: f64,
}

pub(crate) struct AnnealingParams {
    pub(crate) exchange_interval: usize,
    pub(crate) positive_tardiness: AnnealingRegimeParams,
    pub(crate) zero_tardiness: AnnealingRegimeParams,
}

pub(crate) struct AnnealingAttempt<S> {
    pub(crate) neighbor_kind: usize,
    pub(crate) candidate: Option<S>,
    #[cfg(feature = "anneal-visualizer")]
    pub(crate) selected_block_ids: Option<Vec<usize>>,
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

pub(crate) struct AnnealingResult<S, O> {
    pub(crate) best: O,
    pub(crate) worker_bests: Vec<S>,
}

pub(crate) struct WorkerSummary {
    pub(crate) worker_id: usize,
    pub(crate) iterations: usize,
    pub(crate) accepted: usize,
    pub(crate) improved: usize,
    pub(crate) best_imports: usize,
    pub(crate) positive_tardiness_temperature: (f64, f64),
    pub(crate) zero_tardiness_temperature: (f64, f64),
    pub(crate) current_score: f64,
    pub(crate) local_best_score: f64,
    pub(crate) neighbor_stats: Vec<NeighborStats>,
}

pub(crate) trait AnnealingDelegate: Sync {
    type State: AnnealingState;
    type Output;

    fn name(&self) -> &'static str;

    fn neighbor_kinds(&self) -> &'static [&'static str];

    fn propose(
        &self,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State>;

    fn on_shared_best(&self, _state: &Self::State, _timer: Timer) {}

    fn should_stop(&self, _state: &Self::State) -> bool {
        false
    }

    #[cfg(feature = "anneal-visualizer")]
    fn visualizer_schedule<'a>(&self, _state: &'a Self::State) -> Option<&'a [ScheduledBlock]> {
        None
    }

    #[cfg(feature = "anneal-visualizer")]
    fn visualizer_problem(&self) -> Option<&Problem> {
        None
    }

    fn finish(&self, state: Self::State) -> Self::Output;
}

struct WorkerState<S> {
    current: S,
    personal_best: S,
    local_best_score: f64,
    last_imported_revision: Option<u64>,
}

struct WorkerResult<S> {
    summary: WorkerSummary,
    personal_best: S,
}

fn apply_shared_best<S: AnnealingState>(local: &mut WorkerState<S>, best: S, revision: u64) {
    local.current = best;
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

    pub(crate) fn run(
        self,
        initial_states: Vec<D::State>,
        timer: Timer,
    ) -> AnnealingResult<D::State, D::Output> {
        assert_eq!(initial_states.len(), self.worker_count);
        assert!(self.worker_count > 0);
        assert!(self.params.exchange_interval > 0);

        let initial_best = initial_states
            .iter()
            .min_by(|a, b| a.annealing_score().total_cmp(&b.annealing_score()))
            .unwrap()
            .clone();
        let shared = SharedBest::new(initial_best.clone());
        let stop = AtomicBool::new(self.delegate.should_stop(&initial_best));
        let worker_results: Vec<_> = initial_states
            .into_par_iter()
            .enumerate()
            .map(|(worker_id, initial_state)| {
                self.run_worker(&initial_state, &shared, &stop, timer, worker_id)
            })
            .collect();
        let state = shared.into_inner();
        for worker in &worker_results {
            let summary = &worker.summary;
            log!(
                "[{:.4}] [{:8} worker={}] iter={:8}, accepted={:8}, improved={:8}, best_imports={:5}, current={:.3}, local_best={:.3}, temp(z1>0)={:.6}->{:.6}, temp(z1=0)={:.6}->{:.6}\nneighbor stats:\n{}",
                timer.elapsed_seconds(),
                self.delegate.name(),
                summary.worker_id,
                summary.iterations,
                summary.accepted,
                summary.improved,
                summary.best_imports,
                summary.current_score,
                summary.local_best_score,
                summary.positive_tardiness_temperature.0,
                summary.positive_tardiness_temperature.1,
                summary.zero_tardiness_temperature.0,
                summary.zero_tardiness_temperature.1,
                format_neighbor_stats(&summary.neighbor_stats, self.delegate.neighbor_kinds()),
            );
        }
        log!(
            "[{:.4}] [{}] result: best={:.3}",
            timer.elapsed_seconds(),
            self.delegate.name(),
            state.annealing_score(),
        );
        AnnealingResult {
            best: self.delegate.finish(state),
            worker_bests: worker_results
                .into_iter()
                .map(|worker| worker.personal_best)
                .collect(),
        }
    }

    fn run_worker(
        &self,
        initial_state: &D::State,
        shared: &SharedBest<D::State>,
        stop: &AtomicBool,
        timer: Timer,
        worker_id: usize,
    ) -> WorkerResult<D::State> {
        let score = initial_state.annealing_score();
        let mut local = WorkerState {
            current: initial_state.clone(),
            personal_best: initial_state.clone(),
            local_best_score: score,
            last_imported_revision: None,
        };
        let temperature = AnnealingTemperature::new(timer, self.deadline, &self.params);
        let positive_tardiness_temperature = temperature.positive_tardiness.range;
        let zero_tardiness_temperature = temperature.zero_tardiness.range;
        let mut context = AnnealingWorkerContext::new(
            self.deadline,
            self.rng_seed.wrapping_add(worker_id as u64),
        );
        #[cfg(feature = "anneal-visualizer")]
        let mut visualizer = self
            .delegate
            .visualizer_schedule(&local.current)
            .and_then(|_| AnnealVisualizer::new(self.delegate.name(), worker_id));

        let mut accepted = 0usize;
        let mut improved = 0usize;
        let mut best_imports = 0usize;
        let mut neighbor_stats =
            vec![NeighborStats::default(); self.delegate.neighbor_kinds().len()];
        let mut next_status_time = (timer.elapsed_seconds() / STATUS_LOG_INTERVAL_SECONDS).floor()
            * STATUS_LOG_INTERVAL_SECONDS
            + STATUS_LOG_INTERVAL_SECONDS;

        loop {
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let Some(elapsed) = context.next(timer) else {
                break;
            };
            if context.should_exchange(self.params.exchange_interval)
                && self.exchange_shared(&mut local, shared)
            {
                best_imports += 1;
            }

            let current_score = local.current.annealing_score();
            let has_tardiness = local.current.has_tardiness();
            let current_temperature = temperature.current(elapsed, has_tardiness);
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
            let neighbor_kind = attempt.neighbor_kind;
            let stats = &mut neighbor_stats[neighbor_kind];
            stats.selected += 1;

            let Some(candidate) = attempt.candidate else {
                stats.time_sec += neighbor_start.elapsed().as_secs_f64();
                continue;
            };
            stats.succeeded += 1;
            let candidate_score = candidate.annealing_score();
            let delta = candidate_score - current_score;
            let accepted_candidate = candidate_score <= accept_threshold;
            let improved_best =
                accepted_candidate && candidate_score + EPS < local.local_best_score;
            #[cfg(feature = "anneal-visualizer")]
            if let Some(selected_block_ids) = attempt.selected_block_ids.as_deref() {
                self.visualize_snapshot(
                    &mut visualizer,
                    &local.current,
                    &candidate,
                    SnapshotMeta {
                        worker: worker_id,
                        iter: context.iterations(),
                        elapsed,
                        reason: if accepted_candidate {
                            "accepted"
                        } else {
                            "rejected"
                        },
                        neighbor: Some(self.delegate.neighbor_kinds()[neighbor_kind]),
                        accepted: accepted_candidate,
                        improved_current: delta < -EPS,
                        improved_best,
                        score: candidate_score,
                        current_score,
                        best_score: if improved_best {
                            candidate_score
                        } else {
                            local.local_best_score
                        },
                        delta,
                        selected_block_ids,
                    },
                );
            }

            if delta < -EPS {
                stats.improved += 1;
                stats.improved_delta_sum += -delta;
            }
            if !accepted_candidate {
                stats.time_sec += neighbor_start.elapsed().as_secs_f64();
                continue;
            }

            local.current = candidate;
            accepted += 1;
            stats.accepted += 1;

            if improved_best {
                local.personal_best.clone_from(&local.current);
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
            if self.delegate.should_stop(&local.current) {
                stop.store(true, Ordering::Relaxed);
            }

            stats.time_sec += neighbor_start.elapsed().as_secs_f64();
        }

        WorkerResult {
            summary: WorkerSummary {
                worker_id,
                iterations: context.iterations(),
                accepted,
                improved,
                best_imports,
                positive_tardiness_temperature,
                zero_tardiness_temperature,
                current_score: local.current.annealing_score(),
                local_best_score: local.local_best_score,
                neighbor_stats,
            },
            personal_best: local.personal_best,
        }
    }

    #[cfg(feature = "anneal-visualizer")]
    fn visualize_snapshot(
        &self,
        visualizer: &mut Option<AnnealVisualizer>,
        current: &D::State,
        candidate: &D::State,
        meta: SnapshotMeta<'_>,
    ) {
        if let (Some(visualizer), Some(problem), Some(current_schedule), Some(candidate_schedule)) = (
            visualizer,
            self.delegate.visualizer_problem(),
            self.delegate.visualizer_schedule(current),
            self.delegate.visualizer_schedule(candidate),
        ) {
            visualizer.snapshot(meta, problem, current_schedule, candidate_schedule);
        }
    }

    fn exchange_shared(
        &self,
        local: &mut WorkerState<D::State>,
        shared: &SharedBest<D::State>,
    ) -> bool {
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
            apply_shared_best(local, best, revision);
            true
        } else {
            false
        }
    }
}
