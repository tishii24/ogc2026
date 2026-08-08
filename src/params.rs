use std::{fs, path::Path};

use serde::Deserialize;

use crate::{
    Problem,
    solver::annealing::{AnnealingParams, AnnealingRegimeParams},
    utils::random::Random,
};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverParams {
    pub runtime: RuntimeParams,
    pub phases: PhaseParams,
    pub precompute: PrecomputeParams,
    pub insert: InsertParams,
    pub annealing: AnnealingConfigs,
    pub preoptimize: PreoptimizeSolverParams,
    pub neighbor: NeighborParams,
}

impl SolverParams {
    pub fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read_to_string(path)
            .map_err(|err| format!("failed to read params {}: {err}", path.display()))?;
        serde_yaml::from_str(&input)
            .map_err(|err| format!("failed to parse params {}: {err}", path.display()))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeParams {
    pub worker_count: usize,
    pub solver_seed: u64,
    pub preoptimize_seed: u64,
    pub solve_time_buffer_seconds: f64,
    pub solution_emit_min_interval_seconds: f64,
    pub solution_emit_max_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseParams {
    pub initial_preoptimize: LimitedPhaseParams,
    pub optimize: OptimizePhaseParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitedPhaseParams {
    pub time_ratio: f64,
    pub max_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizePhaseParams {
    pub base_horizon_size: usize,
    pub time_allocation_power: f64,
    pub horizon_w2_power: Option<f64>,
    pub expand_time_ratio: f64,
    pub max_expand_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecomputeParams {
    pub orientation_neighbor_limit: usize,
    pub swap_neighbor_area_top_k: usize,
    pub swap_neighbor_align_delta: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsertParams {
    pub y_buffer: usize,
    pub anchor_randomness: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeSolverParams {
    pub alpha: f64,
    pub beta: f64,
    pub bay_padding: f64,
    pub congestion_weight: f64,
    pub initial_build: LimitedPhaseParams,
    pub neighbor_probabilities: PreoptimizeNeighborProbabilities,
    pub neighbor: PreoptimizeNeighborParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeNeighborProbabilities {
    pub relocate: f64,
    pub swap: f64,
    pub large_reconstruct: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeNeighborParams {
    pub remove_count: UsizeRangeDistribution,
    pub bad_block_sample_count: usize,
    pub bad_block_select_probability: f64,
    pub max_relocate_attempts: usize,
    pub max_time_shift: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightScaleConfig {
    #[serde(default)]
    pub w1_scale: Option<f64>,
    #[serde(default)]
    pub w3_scale: Option<f64>,
}

impl WeightScaleConfig {
    fn make(&self, problem: &Problem) -> f64 {
        [
            self.w1_scale.map(|scale| scale * problem.weights.w1),
            self.w3_scale.map(|scale| scale * problem.weights.w3),
        ]
        .into_iter()
        .flatten()
        .reduce(f64::min)
        .unwrap_or(0.0)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingRegimeConfig {
    pub temperature_w1_scale: Option<(f64, f64)>,
    pub temperature_w3_scale: Option<(f64, f64)>,
    pub temperature_initial_score_per_block_scale: Option<(f64, f64)>,
    #[serde(default)]
    pub exchange_threshold: Option<WeightScaleConfig>,
}

impl AnnealingRegimeConfig {
    fn make(
        &self,
        problem: &Problem,
        initial_score: f64,
        block_count: usize,
    ) -> AnnealingRegimeParams {
        let initial_score_per_block = initial_score / block_count.max(1) as f64;
        let candidates = [
            self.temperature_w1_scale
                .map(|scale| [scale.0 * problem.weights.w1, scale.1 * problem.weights.w1]),
            self.temperature_w3_scale
                .map(|scale| [scale.0 * problem.weights.w3, scale.1 * problem.weights.w3]),
            self.temperature_initial_score_per_block_scale.map(|scale| {
                [
                    scale.0 * initial_score_per_block,
                    scale.1 * initial_score_per_block,
                ]
            }),
        ];
        let temperature = [0, 1].map(|index| {
            candidates
                .iter()
                .flatten()
                .map(|range| range[index])
                .reduce(f64::min)
                .unwrap()
        });
        AnnealingRegimeParams {
            temperature: (temperature[0], temperature[1]),
            exchange_threshold: self
                .exchange_threshold
                .as_ref()
                .map(|threshold| threshold.make(problem))
                .unwrap_or(0.0),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingConfigs {
    pub preoptimize: AnnealingParamsConfig,
    pub optimize: AnnealingParamsConfig,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingParamsConfig {
    pub exchange_interval: usize,
    pub positive_tardiness: AnnealingRegimeConfig,
    pub zero_tardiness: AnnealingRegimeConfig,
    #[serde(default)]
    pub worker_temperature_scale: f64,
    pub tabu_capacity: usize,
}

impl AnnealingParamsConfig {
    pub(crate) fn make(
        &self,
        problem: &Problem,
        initial_score: f64,
        block_count: usize,
    ) -> AnnealingParams {
        AnnealingParams {
            exchange_interval: self.exchange_interval,
            positive_tardiness: self
                .positive_tardiness
                .make(problem, initial_score, block_count),
            zero_tardiness: self
                .zero_tardiness
                .make(problem, initial_score, block_count),
            worker_temperature_scale: self.worker_temperature_scale,
            tabu_capacity: self.tabu_capacity,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborParams {
    pub probabilities: NeighborProbabilities,
    pub reconstruct: ReconstructNeighborParams,
    pub shift: ShiftNeighborParams,
    pub rotate: RotateNeighborParams,
    pub swap: SwapNeighborParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructNeighborParams {
    pub remove_count: UsizeRangeDistribution,
    pub remove_pool_factor: usize,
    pub remove_blocks_per_seed: UsizeRangeDistribution,
    pub remove_entry_base_interval_weights: Vec<f64>,
    pub remove_entry_seed_candidate_count: usize,
    pub remove_seed_method_weights: RemoveSeedMethodWeights,
    pub remove_fluidity_slack_weight_range: (f64, f64),
    pub remove_fluidity_pref_spread_weight_range: (f64, f64),
    pub remove_x_distance_weight_range: (f64, f64),
    pub remove_y_distance_weight_range: (f64, f64),
    pub remove_t_distance_weight_range: (f64, f64),
    pub remove_distance_power_range: (f64, f64),
    pub workload_weight_range: (f64, f64),
    pub volume_weight_range: (f64, f64),
    pub pref_spread_weight_range: (f64, f64),
    pub limit_time_urgency_weight_range: (f64, f64),
    pub release_time_weight_range: (f64, f64),
    pub slack_tightness_weight_range: (f64, f64),
    pub current_penalty_weight_range: (f64, f64),
    pub order_random_weight_range: (f64, f64),
    pub insert_candidate_top_k: usize,
    pub insert_candidate_select_p: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UsizeRangeDistributionType {
    Lower,
    Centered,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsizeRangeDistribution {
    #[serde(rename = "type")]
    pub distribution_type: UsizeRangeDistributionType,
    pub range: (usize, usize),
    pub shape: f64,
}

impl UsizeRangeDistribution {
    pub fn sample(&self, rng: &mut impl Random) -> usize {
        match self.distribution_type {
            UsizeRangeDistributionType::Lower => {
                rng.gen_range_lower(self.range.0, self.range.1 + 1, self.shape)
            }
            UsizeRangeDistributionType::Centered => {
                rng.gen_range_centered(self.range.0, self.range.1 + 1, self.shape)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShiftNeighborParams {
    pub dy_range: (i64, i64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RotateNeighborParams {
    pub dy_range: (i64, i64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SwapNeighborParams {
    pub neighbor_top_k: usize,
    pub dy_range: (i64, i64),
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoveSeedMethodWeights {
    pub badness: f64,
    pub fluidity: f64,
    pub random: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborProbabilities {
    pub large_reconstruct: f64,
    pub shift: f64,
    #[serde(rename = "move")]
    pub move_block: f64,
    pub rotate: f64,
    pub swap: f64,
}

impl NeighborProbabilities {
    pub(crate) fn weights(&self) -> [f64; 5] {
        [
            self.large_reconstruct,
            self.shift,
            self.move_block,
            self.rotate,
            self.swap,
        ]
    }
}
