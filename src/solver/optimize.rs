use std::ops::Range;

use crate::{
    EPS, INF, Problem, ScheduledBlock,
    params::{
        AnnealingParamsConfig, InsertParams, NeighborParams, OptimizePhaseParams,
        ReconstructNeighborParams,
    },
    utils::{random::RandPcg64Mcg, time::Timer},
};

use super::{
    PreoptimizeState,
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    insert::insert_greedy,
    neighbors::{
        NeighborKind, sample_neighbor, try_move_neighbor, try_rotate_neighbor, try_shift_neighbor,
        try_swap_neighbor,
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

    fn tabu_key(&self) -> Option<u64> {
        Some(hash_schedule(&self.schedule))
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
        initial: OptimizeState,
        deadline: f64,
        annealing: &AnnealingParamsConfig,
        neighbor_params: &NeighborParams,
        insert_params: &InsertParams,
        w2: f64,
        candidate_emitter: Option<&CandidateEmitter>,
        seed: u64,
        max_worker_count: usize,
    ) -> OptimizeState {
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        let annealing_params = annealing.make(
            self.problem,
            initial.annealing_score(),
            initial.schedule.len(),
        );
        let delegate = OptimizeAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            w2,
            initial,
            params: neighbor_params,
            insert_params,
            timer: self.timer,
            candidate_emitter,
        };
        Annealer::new(deadline, worker_count, seed, annealing_params, delegate).run(self.timer)
    }
}

struct OptimizeAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    w2: f64,
    initial: OptimizeState,
    params: &'a NeighborParams,
    insert_params: &'a InsertParams,
    timer: Timer,
    candidate_emitter: Option<&'a CandidateEmitter>,
}

impl AnnealingDelegate for OptimizeAnnealingDelegate<'_> {
    type State = OptimizeState;
    type Output = OptimizeState;

    fn initial_state(&self) -> Self::State {
        self.initial.clone()
    }

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
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let params = self.params;
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
                self.w2,
            ),
            NeighborKind::Rotate => try_rotate_neighbor(
                self.problem,
                self.pre,
                &current.schedule,
                rng,
                &params.rotate,
            ),
            NeighborKind::Swap => {
                try_swap_neighbor(self.problem, self.pre, &current.schedule, rng, &params.swap)
            }
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
    let mut state = make_optimize_state(problem, pre, Vec::new(), 0.0);
    let mut rng = RandPcg64Mcg::new(seed);
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
        state = extend_schedule(
            problem,
            pre,
            preopt,
            state,
            &order[horizon.clone()],
            insert_params,
            &neighbor_params.reconstruct,
            w2,
            timer,
            now + expand_duration,
            &mut rng,
        );
        log!(
            "[{:.4}] [optimize] horizon={}/{}, blocks={}, initial={:.3}, w2={:.3}",
            timer.elapsed_seconds(),
            horizon_index + 1,
            horizons.len(),
            horizon.end,
            state.objective,
            w2,
        );

        if is_last {
            candidate_emitter.emit(&state, timer, true);
        }
        if state.objective <= EPS {
            continue;
        }
        if timer.elapsed_seconds() >= horizon_deadline {
            continue;
        }
        state = annealer.run(
            state,
            horizon_deadline,
            annealing_params,
            neighbor_params,
            insert_params,
            w2,
            is_last.then_some(candidate_emitter),
            seed.wrapping_add(((horizon_index + 1) as u64) << 32),
            max_worker_count,
        );
    }

    make_optimize_state(problem, pre, state.schedule, problem.weights.w2)
}

fn hash_schedule(schedule: &[ScheduledBlock]) -> u64 {
    let mut hash = mix_hash(1469598103934665603, schedule.len() as u64);
    for &block in schedule {
        hash ^= hash_scheduled_block(block);
    }
    hash
}

fn hash_scheduled_block(block: ScheduledBlock) -> u64 {
    let mut hash = 1469598103934665603;
    hash = mix_hash(hash, block.block_id as u64);
    hash = mix_hash(hash, block.bay_id as u64);
    hash = mix_hash(hash, block.orient_idx as u64);
    hash = mix_hash(hash, block.x as u64);
    hash = mix_hash(hash, block.y as u64);
    hash = mix_hash(hash, block.entry_time as u64);
    mix_hash(hash, block.exit_time as u64)
}

fn mix_hash(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash.wrapping_mul(1099511628211)
}
