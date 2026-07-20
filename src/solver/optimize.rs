use super::*;
use crate::{
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    params::{GlobalOptimizeParams, InsertParams},
    solver_util::sample_neighbor,
};

impl AnnealingState for OptimizeState {
    fn annealing_score(&self) -> f64 {
        self.score
    }

    fn tabu_key(&self) -> Option<u64> {
        Some(hash_schedule(&self.blocks))
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
        precedence_margin: i64,
        timer: Timer,
    ) -> Self {
        Self {
            problem,
            pre,
            constraints: build_precedence_constraints(problem, abstract_state, precedence_margin),
            timer,
        }
    }

    pub fn run(
        &self,
        initial: OptimizeState,
        deadline: f64,
        params: &GlobalOptimizeParams,
        neighbor_params: &NeighborParams,
        insert_params: &InsertParams,
        constrained: bool,
        seed: u64,
        max_worker_count: usize,
    ) -> OptimizeState {
        let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
        let delegate = GlobalAnnealingDelegate {
            problem: self.problem,
            pre: self.pre,
            constraints: constrained.then_some(&self.constraints),
            name: if constrained { "global-c" } else { "global" },
            initial,
            exchange_threshold_w1_scale: params.exchange_threshold_w1_scale,
            params: neighbor_params,
            insert_params,
        };
        Annealer::new(
            deadline,
            worker_count,
            seed,
            params.annealing.make(),
            delegate,
        )
        .run(self.timer)
    }
}

struct GlobalAnnealingDelegate<'a> {
    problem: &'a Problem,
    pre: &'a Precompute,
    constraints: Option<&'a PrecedenceConstraints>,
    name: &'static str,
    initial: OptimizeState,
    exchange_threshold_w1_scale: f64,
    params: &'a NeighborParams,
    insert_params: &'a InsertParams,
}

impl AnnealingDelegate for GlobalAnnealingDelegate<'_> {
    type State = OptimizeState;
    type Output = OptimizeState;

    fn initial_states(&self) -> Vec<Self::State> {
        vec![self.initial.clone()]
    }

    fn name(&self) -> &'static str {
        self.name
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
        let constraints = self.constraints;
        let params = self.params;
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
                self.insert_params,
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
                self.insert_params,
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
