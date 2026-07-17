use super::*;
use crate::{
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    params::{BayOptimizeParams, GlobalOptimizeParams},
    solver_util::sample_neighbor,
};

#[derive(Clone)]
pub struct BayOptimizeState {
    pub bay_id: usize,
    pub score: f64,
    pub tardiness: i64,
    pub blocks: Vec<ScheduledBlock>,
}

impl AnnealingState for BayOptimizeState {
    fn annealing_score(&self) -> f64 {
        self.score
    }

    fn tabu_key(&self) -> Option<u64> {
        Some(hash_schedule(&self.blocks))
    }
}

impl AnnealingState for OptimizeState {
    fn annealing_score(&self) -> f64 {
        self.score
    }

    fn tabu_key(&self) -> Option<u64> {
        Some(hash_schedule(&self.blocks))
    }
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

    pub fn run(
        &self,
        initial: OptimizeState,
        deadline: f64,
        params: &BayOptimizeParams,
        neighbor_params: &NeighborParams,
        seed: u64,
        max_worker_count: usize,
    ) -> OptimizeState {
        if self.timer.elapsed_seconds() >= deadline {
            return initial;
        }

        let mut blocks_by_bay = vec![Vec::new(); self.problem.bays.len()];
        for block in initial.blocks {
            blocks_by_bay[block.bay_id].push(block);
        }
        let initial_states = blocks_by_bay
            .into_iter()
            .enumerate()
            .map(|(bay_id, blocks)| {
                let tardiness = bay_tardiness(self.problem, &blocks);
                BayOptimizeState {
                    bay_id,
                    score: self.problem.weights.w1 * tardiness as f64,
                    tardiness,
                    blocks,
                }
            })
            .collect();
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        let delegate = BayAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            constraints: &self.constraints,
            target_tardiness: &self.target_tardiness,
            initial_states,
            exchange_threshold_w1_scale: params.exchange_threshold_w1_scale,
            params: neighbor_params,
        };
        Annealer::new(
            deadline,
            worker_count,
            seed,
            params.annealing.make(self.problem),
            delegate,
        )
        .run(self.timer)
    }
}

struct BayAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: &'a PrecedenceConstraints,
    target_tardiness: &'a [i64],
    initial_states: Vec<BayOptimizeState>,
    exchange_threshold_w1_scale: f64,
    params: &'a NeighborParams,
}

impl AnnealingDelegate for BayAnnealingDelegate<'_> {
    type State = BayOptimizeState;
    type Output = OptimizeState;

    fn initial_states(&self) -> Vec<Self::State> {
        self.initial_states.clone()
    }

    fn name(&self) -> &'static str {
        "bay"
    }

    fn neighbor_kinds(&self) -> &'static [&'static str] {
        NEIGHBOR_KINDS
    }

    fn exchange_threshold(&self, _domain: usize) -> f64 {
        self.exchange_threshold_w1_scale * self.problem.weights.w1
    }

    fn propose(
        &self,
        domain: usize,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        debug_assert_eq!(domain, current.bay_id);
        let probabilities = self.params.probabilities();
        let neighbor = sample_neighbor(rng, &probabilities);
        let blocks = match neighbor {
            NeighborKind::LargeReconstruct => try_bay_large_reconstruct(
                self.problem,
                self.pre,
                self.constraints,
                &current.blocks,
                rng,
                accept_threshold,
                current.bay_id,
                self.params,
            ),
            NeighborKind::Shift => try_shift_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                Some(self.constraints),
                self.params,
            ),
            NeighborKind::Move => try_move_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                Some(self.constraints),
                Some(current.bay_id),
                self.params,
            ),
            NeighborKind::Rotate => try_rotate_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                Some(self.constraints),
                self.params,
            ),
            NeighborKind::Swap => try_swap_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                Some(self.constraints),
                true,
                self.params,
            ),
        };
        AnnealingAttempt {
            neighbor_kind: neighbor.index(),
            candidate: blocks.map(|blocks| {
                let tardiness = bay_tardiness(self.problem, &blocks);
                BayOptimizeState {
                    bay_id: current.bay_id,
                    score: self.problem.weights.w1 * tardiness as f64,
                    tardiness,
                    blocks,
                }
            }),
        }
    }

    fn is_finished(&self, domain: usize, state: &Self::State) -> bool {
        state.tardiness <= self.target_tardiness[domain]
    }

    fn finish(&self, states: Vec<Self::State>) -> Self::Output {
        let mut blocks = Vec::with_capacity(self.problem.blocks.len());
        for state in states {
            blocks.extend(state.blocks);
        }
        let score = score_schedule(self.problem, self.pre, &blocks);
        OptimizeState { score, blocks }
    }
}

pub struct GlobalAnnealing<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: PrecedenceConstraints,
    timer: Timer,
}

impl<'a> GlobalAnnealing<'a> {
    pub fn new(
        problem: &'a Problem,
        pre: &'a Precompute,
        abstract_state: &PreoptimizeState,
        timer: Timer,
    ) -> Self {
        Self {
            problem,
            pre,
            constraints: build_precedence_constraints(problem, abstract_state),
            timer,
        }
    }

    pub fn run(
        &self,
        initial: OptimizeState,
        deadline: f64,
        params: &GlobalOptimizeParams,
        neighbor_params: &NeighborParams,
        constrained_neighbor_params: &NeighborParams,
        seed: u64,
        max_worker_count: usize,
    ) -> OptimizeState {
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        let start_time = self.timer.elapsed_seconds();
        let delegate = GlobalAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            constraints: &self.constraints,
            timer: self.timer,
            constraint_deadline: start_time
                + (deadline - start_time).max(0.0) * params.constraint_time_ratio,
            initial,
            exchange_threshold_w1_scale: params.exchange_threshold_w1_scale,
            constrained_neighbor_params,
            params: neighbor_params,
        };
        Annealer::new(
            deadline,
            worker_count,
            seed,
            params.annealing.make(self.problem),
            delegate,
        )
        .run(self.timer)
    }
}

struct GlobalAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: &'a PrecedenceConstraints,
    timer: Timer,
    constraint_deadline: f64,
    initial: OptimizeState,
    exchange_threshold_w1_scale: f64,
    constrained_neighbor_params: &'a NeighborParams,
    params: &'a NeighborParams,
}

impl AnnealingDelegate for GlobalAnnealingDelegate<'_> {
    type State = OptimizeState;
    type Output = OptimizeState;

    fn initial_states(&self) -> Vec<Self::State> {
        vec![self.initial.clone()]
    }

    fn name(&self) -> &'static str {
        "global"
    }

    fn neighbor_kinds(&self) -> &'static [&'static str] {
        NEIGHBOR_KINDS
    }

    fn exchange_threshold(&self, _domain: usize) -> f64 {
        self.exchange_threshold_w1_scale * self.problem.weights.w1
    }

    fn is_finished(&self, _domain: usize, _state: &Self::State) -> bool {
        false
    }

    fn propose(
        &self,
        _domain: usize,
        current: &Self::State,
        accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let constraints =
            (self.timer.elapsed_seconds() < self.constraint_deadline).then_some(self.constraints);
        let params = if constraints.is_some() {
            self.constrained_neighbor_params
        } else {
            self.params
        };
        let probabilities = params.probabilities();
        let neighbor = sample_neighbor(rng, &probabilities);
        let blocks = match neighbor {
            NeighborKind::LargeReconstruct => try_large_reconstruct(
                self.problem,
                self.pre,
                constraints,
                &current.blocks,
                rng,
                accept_threshold,
                params,
            ),
            NeighborKind::Shift => try_shift_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                params,
            ),
            NeighborKind::Move => try_move_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                None,
                params,
            ),
            NeighborKind::Rotate => try_rotate_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                params,
            ),
            NeighborKind::Swap => try_swap_neighbor(
                self.problem,
                self.pre,
                &current.blocks,
                rng,
                constraints,
                constraints.is_some(),
                params,
            ),
        };
        AnnealingAttempt {
            neighbor_kind: neighbor.index(),
            candidate: blocks.map(|blocks| OptimizeState {
                score: score_schedule(self.problem, self.pre, &blocks),
                blocks,
            }),
        }
    }

    fn finish(&self, mut states: Vec<Self::State>) -> Self::Output {
        states.pop().unwrap()
    }
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
