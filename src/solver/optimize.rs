use std::ops::Range;

use rayon::prelude::*;

use crate::{
    EPS, INF, Problem, ScheduledBlock,
    params::{
        AnnealingParamsConfig, InsertAnchor, InsertParams, NeighborParams, OptimizePhaseParams,
        ReconstructNeighborParams,
    },
    utils::{random::RandPcg64Mcg, time::Timer},
};

use super::{
    PreoptimizeState,
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingResult, AnnealingState},
    insert::{insert_greedy, sample_insert_anchor},
    neighbors::{
        NeighborKind, sample_neighbor, try_move_neighbor, try_rotate_neighbor, try_shift_neighbor,
    },
    objective::{ScheduleScore, score_schedule, score13_block},
    output::CandidateEmitter,
    precompute::Precompute,
    reconstruct::{sort_default_reconstruct_order, try_large_reconstruct},
};

#[derive(Clone, Debug)]
pub(super) struct OptimizeState {
    pub(super) objective: f64,
    pub(super) total_tardiness: i64,
    pub(super) schedule: Vec<ScheduledBlock>,
}

impl AnnealingState for OptimizeState {
    fn annealing_score(&self) -> f64 {
        self.objective
    }

    fn has_tardiness(&self) -> bool {
        self.total_tardiness > 0
    }
}

pub(super) fn make_optimize_state(
    problem: &Problem,
    pre: &Precompute,
    schedule: Vec<ScheduledBlock>,
    w2: f64,
) -> OptimizeState {
    let ScheduleScore {
        objective,
        total_tardiness,
    } = score_schedule(problem, pre, &schedule, w2);
    OptimizeState {
        objective,
        total_tardiness,
        schedule,
    }
}

pub(super) struct OptimizeAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    timer: Timer,
}

impl<'a> OptimizeAnnealing<'a> {
    pub(super) fn new(problem: &'a Problem, pre: &'a Precompute, timer: Timer) -> Self {
        Self {
            problem,
            pre,
            timer,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        &self,
        initial_states: Vec<OptimizeState>,
        deadline: f64,
        annealing: &AnnealingParamsConfig,
        neighbor_params: &NeighborParams,
        insert_params: &InsertParams,
        w2: f64,
        candidate_emitter: Option<&CandidateEmitter>,
        seed: u64,
        max_worker_count: usize,
    ) -> AnnealingResult<OptimizeState, OptimizeState> {
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        assert_eq!(initial_states.len(), worker_count);
        let initial = initial_states
            .iter()
            .min_by(|a, b| a.objective.total_cmp(&b.objective))
            .unwrap();
        let annealing_params = annealing.make(
            self.problem,
            initial.annealing_score(),
            initial.schedule.len(),
        );
        let delegate = OptimizeAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            w2,
            params: neighbor_params,
            insert_params,
            timer: self.timer,
            candidate_emitter,
        };
        Annealer::new(deadline, worker_count, seed, annealing_params, delegate)
            .run(initial_states, self.timer)
    }
}

struct OptimizeAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    w2: f64,

    params: &'a NeighborParams,
    insert_params: &'a InsertParams,
    timer: Timer,
    candidate_emitter: Option<&'a CandidateEmitter>,
}

impl AnnealingDelegate for OptimizeAnnealingDelegate<'_> {
    type State = OptimizeState;
    type Output = OptimizeState;

    fn name(&self) -> &'static str {
        "optimize"
    }

    fn neighbor_kinds(&self) -> &'static [&'static str] {
        NeighborKind::NAMES
    }

    fn on_shared_best(&self, state: &Self::State, _timer: Timer) {
        if let Some(candidate_emitter) = self.candidate_emitter {
            candidate_emitter.emit(state, self.timer, false);
        }
    }

    fn should_stop(&self, state: &Self::State) -> bool {
        state.objective <= EPS
    }

    #[cfg(feature = "anneal-visualizer")]
    fn visualizer_schedule<'a>(&self, state: &'a Self::State) -> Option<&'a [ScheduledBlock]> {
        Some(&state.schedule)
    }

    #[cfg(feature = "anneal-visualizer")]
    fn visualizer_problem(&self) -> Option<&Problem> {
        Some(self.problem)
    }

    fn propose(
        &self,
        worker_id: usize,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let params = self.params;
        let primary_anchor = self.insert_params.worker_primary_anchor(worker_id);
        let probabilities = params.probabilities.weights();
        let neighbor = sample_neighbor(rng, &probabilities);
        #[cfg(feature = "anneal-visualizer")]
        let mut selected_block_ids = None;
        let schedule = match neighbor {
            NeighborKind::LargeReconstruct => try_large_reconstruct(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                accept_threshold,
                &params.reconstruct,
                self.insert_params,
                primary_anchor,
                self.w2,
            )
            .map(|result| {
                #[cfg(feature = "anneal-visualizer")]
                {
                    selected_block_ids = Some(result.selected_block_ids);
                }
                result.schedule
            }),
            NeighborKind::Shift => try_shift_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                &params.shift,
            ),
            NeighborKind::Move => try_move_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                self.insert_params,
                primary_anchor,
                self.w2,
            ),
            NeighborKind::Rotate => try_rotate_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                &params.rotate,
            ),
        };
        AnnealingAttempt {
            neighbor_kind: neighbor.index(),
            candidate: schedule
                .map(|schedule| make_optimize_state(self.problem, self.pre, schedule, self.w2)),
            #[cfg(feature = "anneal-visualizer")]
            selected_block_ids,
        }
    }

    fn finish(&self, state: Self::State) -> Self::Output {
        state
    }
}

fn build_admission_order(
    problem: &Problem,
    pre: &Precompute,
    preopt: &PreoptimizeState,
) -> Vec<usize> {
    let mut order: Vec<_> = (0..problem.blocks.len()).collect();
    order.sort_by(|&a, &b| {
        let volume_a = pre.max_footprint_area[a] * problem.blocks[a].processing_time as f64;
        let volume_b = pre.max_footprint_area[b] * problem.blocks[b].processing_time as f64;
        preopt.blocks[a]
            .entry_time
            .cmp(&preopt.blocks[b].entry_time)
            .then_with(|| volume_b.total_cmp(&volume_a))
            .then(a.cmp(&b))
    });
    order
}

fn build_horizons(order: &[usize], base_horizon_size: usize) -> Vec<Range<usize>> {
    let horizon_count = order.len().div_ceil(base_horizon_size);
    (0..horizon_count)
        .map(|index| order.len() * index / horizon_count..order.len() * (index + 1) / horizon_count)
        .collect()
}

fn horizon_w2(w2: f64, progress: f64, is_last: bool, power: Option<f64>) -> f64 {
    match power {
        Some(power) => w2 * progress.powf(power),
        None => {
            if is_last {
                w2
            } else {
                0.0
            }
        }
    }
}

fn extend_schedule(
    problem: &Problem,
    pre: &Precompute,
    preopt: &PreoptimizeState,
    state: OptimizeState,
    added_block_ids: &[usize],
    insert_params: &InsertParams,
    primary_anchor: InsertAnchor,
    reconstruct_params: &ReconstructNeighborParams,
    w2: f64,
    timer: Timer,
    deadline: f64,
    rng: &mut RandPcg64Mcg,
) -> OptimizeState {
    let base_schedule = state.schedule;
    let mut best = None;
    let mut trial_count = 0;

    'trial: loop {
        if best.is_some() && timer.elapsed_seconds() >= deadline {
            break;
        }
        trial_count += 1;

        let anchor = sample_insert_anchor(rng, insert_params, primary_anchor);
        let mut schedule = base_schedule.clone();
        let mut loads = vec![0.0; problem.bays.len()];
        let mut fixed_score13 = 0.0;
        for &scheduled in &schedule {
            loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
            fixed_score13 += score13_block(problem, pre, scheduled);
        }
        let mut order = added_block_ids.to_vec();
        sort_default_reconstruct_order(
            problem,
            &pre.max_footprint_area,
            &pre.pref_spread,
            &mut order,
            rng,
            reconstruct_params,
        );

        for block_id in order {
            if best
                .as_ref()
                .is_some_and(|best: &OptimizeState| fixed_score13 >= best.objective - EPS)
            {
                continue 'trial;
            }
            let block = &problem.blocks[block_id];
            let entry_time = preopt.blocks[block_id].entry_time.max(block.release_time);
            let original = ScheduledBlock {
                block_id,
                bay_id: preopt.blocks[block_id].bay_id,
                orient_idx: 0,
                x: 0,
                y: 0,
                entry_time,
                exit_time: entry_time + block.processing_time,
            };
            let Some(scheduled) = insert_greedy(
                problem,
                pre,
                original,
                block.release_time,
                INF,
                &schedule,
                &loads,
                insert_params,
                &pre.bay_order_by_pref[block_id],
                1,
                1.0,
                w2,
                anchor,
                rng,
            ) else {
                continue 'trial;
            };
            loads[scheduled.bay_id] += block.workload as f64;
            fixed_score13 += score13_block(problem, pre, scheduled);
            schedule.push(scheduled);
        }

        let candidate = make_optimize_state(problem, pre, schedule, w2);
        if best
            .as_ref()
            .is_none_or(|best: &OptimizeState| candidate.objective < best.objective)
        {
            log!(
                "[{:.4}] [expand] best: trial={}, score={:.3}",
                timer.elapsed_seconds(),
                trial_count,
                candidate.objective,
            );
            best = Some(candidate);
        }
        if best.as_ref().unwrap().objective <= EPS {
            break;
        }
    }

    let best = best.unwrap();
    log!(
        "[{:.4}] [expand] finish: trials={}, score={:.3}",
        timer.elapsed_seconds(),
        trial_count,
        best.objective,
    );
    best
}

#[allow(clippy::too_many_arguments)]
pub(super) fn optimize(
    problem: &Problem,
    pre: &Precompute,
    preopt: &PreoptimizeState,
    deadline: f64,
    timer: Timer,
    phase_params: &OptimizePhaseParams,
    annealing_params: &AnnealingParamsConfig,
    neighbor_params: &NeighborParams,
    insert_params: &InsertParams,
    candidate_emitter: &CandidateEmitter,
    max_worker_count: usize,
    seed: u64,
) -> OptimizeState {
    let order = build_admission_order(problem, pre, preopt);
    let horizons = build_horizons(&order, phase_params.base_horizon_size);
    let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
    let initial = make_optimize_state(problem, pre, Vec::new(), 0.0);
    let mut worker_states = vec![initial.clone(); worker_count];
    let mut best_state = initial;
    let annealer = OptimizeAnnealing::new(problem, pre, timer);

    for (horizon_index, horizon) in horizons.iter().enumerate() {
        let is_last = horizon_index + 1 == horizons.len();
        let horizon_progress = horizon.end as f64 / order.len() as f64;
        let w2 = horizon_w2(
            problem.weights.w2,
            horizon_progress,
            is_last,
            phase_params.horizon_w2_power,
        );
        let now = timer.elapsed_seconds();
        let remaining_weight: f64 = horizons[horizon_index..]
            .iter()
            .map(|remaining| (remaining.end as f64).powf(phase_params.time_allocation_power))
            .sum();
        let current_weight = (horizon.end as f64).powf(phase_params.time_allocation_power);
        let horizon_deadline = now + (deadline - now).max(0.0) * current_weight / remaining_weight;
        if is_last {
            candidate_emitter.configure(horizon_deadline - now, timer);
        }
        let expand_duration = ((horizon_deadline - now) * phase_params.expand_time_ratio)
            .min(phase_params.max_expand_seconds);
        let horizon_seed = seed.wrapping_add(((horizon_index + 1) as u64) << 32);
        worker_states = worker_states
            .into_par_iter()
            .enumerate()
            .map(|(worker_id, state)| {
                let mut rng = RandPcg64Mcg::new(horizon_seed.wrapping_add(worker_id as u64));
                let primary_anchor = insert_params.worker_primary_anchor(worker_id);
                extend_schedule(
                    problem,
                    pre,
                    preopt,
                    state,
                    &order[horizon.clone()],
                    insert_params,
                    primary_anchor,
                    &neighbor_params.reconstruct,
                    w2,
                    timer,
                    now + expand_duration,
                    &mut rng,
                )
            })
            .collect();
        best_state = worker_states
            .iter()
            .min_by(|a, b| a.objective.total_cmp(&b.objective))
            .unwrap()
            .clone();
        log!(
            "[{:.4}] [optimize] horizon={}/{}, blocks={}, initial={:.3}, w2={:.3}",
            timer.elapsed_seconds(),
            horizon_index + 1,
            horizons.len(),
            horizon.end,
            best_state.objective,
            w2,
        );

        if is_last {
            candidate_emitter.emit(&best_state, timer, true);
        }
        if best_state.objective <= EPS {
            continue;
        }
        if timer.elapsed_seconds() >= horizon_deadline {
            continue;
        }
        let result = annealer.run(
            worker_states,
            horizon_deadline,
            annealing_params,
            neighbor_params,
            insert_params,
            w2,
            is_last.then_some(candidate_emitter),
            horizon_seed,
            max_worker_count,
        );
        best_state = result.best;
        worker_states = result.worker_bests;
    }

    make_optimize_state(problem, pre, best_state.schedule, problem.weights.w2)
}
