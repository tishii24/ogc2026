use std::{fs, path::Path};

use serde::Deserialize;

use crate::{Problem, annealing::AnnealingParams, solver_util::NeighborKind};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverParamsFile {
    pub runtime: RuntimeParams,
    pub phases: PhaseParams,
    pub precompute: PrecomputeParams,
    pub insert: InsertParams,
    pub annealing: AnnealingParamsConfig,
    pub preoptimize: PreoptimizeSolverParams,
    pub global_optimize: GlobalOptimizeParams,
    pub optimize_neighbor: OptimizeNeighborParams,
}

#[derive(Clone, Debug)]
pub struct SolverParams {
    pub runtime: RuntimeParams,
    pub phases: PhaseParams,
    pub precompute: PrecomputeParams,
    pub insert: InsertParams,
    pub annealing: AnnealingParamsConfig,
    pub preoptimize: PreoptimizeSolverParams,
    pub global_optimize: GlobalOptimizeParams,
    pub global_neighbor: NeighborParams,
}

impl SolverParams {
    pub fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read_to_string(path)
            .map_err(|err| format!("failed to read params {}: {err}", path.display()))?;
        let file: SolverParamsFile = serde_yaml::from_str(&input)
            .map_err(|err| format!("failed to parse params {}: {err}", path.display()))?;
        file.resolve()
    }
}

impl SolverParamsFile {
    fn resolve(self) -> Result<SolverParams, String> {
        let global_neighbor = self
            .optimize_neighbor
            .default
            .with_override(&self.optimize_neighbor.global);
        let params = SolverParams {
            runtime: self.runtime,
            phases: self.phases,
            precompute: self.precompute,
            insert: self.insert,
            annealing: self.annealing,
            preoptimize: self.preoptimize,
            global_optimize: self.global_optimize,
            global_neighbor,
        };
        Ok(params)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeParams {
    pub worker_count: usize,
    pub solver_seed: u64,
    pub preoptimize_seed: u64,
    pub local_search_time_buffer_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseParams {
    pub initial_preoptimize: LimitedPhaseParams,
    pub initial_build: LimitedPhaseParams,
    pub global_constrained: LimitedPhaseParams,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LimitedPhaseParams {
    pub time_ratio: f64,
    pub max_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrecomputeParams {
    pub orientation_neighbor_limit: usize,
    pub other_block_neighbor_area_top_k: usize,
    pub other_block_neighbor_align_delta: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InsertParams {
    pub y_sample_ratio_base: f64,
    pub y_sample_ratio_min: f64,
    pub y_buffer: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreoptimizeSolverParams {
    pub alpha: f64,
    pub beta: f64,
    pub bay_padding: f64,
    pub precedence_margin: i64,
    pub congestion_weight: f64,
    pub neighbor_probabilities: PreoptimizeNeighborProbabilities,
    pub neighbor: PreoptimizeNeighborParams,
    #[serde(default)]
    pub annealing: AnnealingParamsOverride,
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
    pub min_removed_blocks: usize,
    pub max_removed_blocks: usize,
    pub remove_count_sample_power: f64,
    pub remove_seed_per_block: usize,
    pub bad_block_sample_count: usize,
    pub bad_block_select_probability: f64,
    pub max_relocate_attempts: usize,
    pub max_time_shift: i64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlobalOptimizeParams {
    #[serde(default)]
    pub annealing: AnnealingParamsOverride,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightScaleConfig {
    pub w1_scale: f64,
    pub w3_scale: f64,
}

impl WeightScaleConfig {
    fn make(&self, problem: &Problem) -> f64 {
        (self.w1_scale * problem.weights.w1).min(self.w3_scale * problem.weights.w3)
    }

    fn with_override(&self, value: &WeightScaleOverride) -> Self {
        Self {
            w1_scale: value.w1_scale.unwrap_or(self.w1_scale),
            w3_scale: value.w3_scale.unwrap_or(self.w3_scale),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WeightScaleOverride {
    pub w1_scale: Option<f64>,
    pub w3_scale: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingRegimeConfig {
    pub temperature_w1_scale: (f64, f64),
    pub temperature_w3_scale: (f64, f64),
    pub temperature_absolute: (f64, f64),
    pub exchange_threshold: WeightScaleConfig,
}

impl AnnealingRegimeConfig {
    fn make(&self, problem: &Problem) -> crate::annealing::AnnealingRegimeParams {
        crate::annealing::AnnealingRegimeParams {
            temperature: (
                (self.temperature_w1_scale.0 * problem.weights.w1)
                    .min(self.temperature_w3_scale.0 * problem.weights.w3)
                    .min(self.temperature_absolute.0),
                (self.temperature_w1_scale.1 * problem.weights.w1)
                    .min(self.temperature_w3_scale.1 * problem.weights.w3)
                    .min(self.temperature_absolute.1),
            ),
            exchange_threshold: self.exchange_threshold.make(problem),
        }
    }

    fn with_override(&self, value: &AnnealingRegimeOverride) -> Self {
        Self {
            temperature_w1_scale: value
                .temperature_w1_scale
                .unwrap_or(self.temperature_w1_scale),
            temperature_w3_scale: value
                .temperature_w3_scale
                .unwrap_or(self.temperature_w3_scale),
            temperature_absolute: value
                .temperature_absolute
                .unwrap_or(self.temperature_absolute),
            exchange_threshold: self
                .exchange_threshold
                .with_override(&value.exchange_threshold),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnnealingRegimeOverride {
    pub temperature_w1_scale: Option<(f64, f64)>,
    pub temperature_w3_scale: Option<(f64, f64)>,
    pub temperature_absolute: Option<(f64, f64)>,
    pub exchange_threshold: WeightScaleOverride,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnealingParamsConfig {
    pub exchange_interval: usize,
    pub positive_tardiness: AnnealingRegimeConfig,
    pub zero_tardiness: AnnealingRegimeConfig,
    pub worker_temperature_scale: f64,
    pub tabu_capacity: usize,
}

impl AnnealingParamsConfig {
    pub(crate) fn make(&self, problem: &Problem) -> AnnealingParams {
        AnnealingParams {
            exchange_interval: self.exchange_interval,
            positive_tardiness: self.positive_tardiness.make(problem),
            zero_tardiness: self.zero_tardiness.make(problem),
            worker_temperature_scale: self.worker_temperature_scale,
            tabu_capacity: self.tabu_capacity,
        }
    }

    pub(crate) fn with_override(&self, value: &AnnealingParamsOverride) -> Self {
        Self {
            exchange_interval: value.exchange_interval.unwrap_or(self.exchange_interval),
            positive_tardiness: self
                .positive_tardiness
                .with_override(&value.positive_tardiness),
            zero_tardiness: self.zero_tardiness.with_override(&value.zero_tardiness),
            worker_temperature_scale: value
                .worker_temperature_scale
                .unwrap_or(self.worker_temperature_scale),
            tabu_capacity: value.tabu_capacity.unwrap_or(self.tabu_capacity),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnnealingParamsOverride {
    pub exchange_interval: Option<usize>,
    pub positive_tardiness: AnnealingRegimeOverride,
    pub zero_tardiness: AnnealingRegimeOverride,
    pub worker_temperature_scale: Option<f64>,
    pub tabu_capacity: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OptimizeNeighborParams {
    pub default: NeighborParams,
    #[serde(default)]
    pub global: NeighborParamsOverride,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborParams {
    pub probabilities: NeighborProbabilities,
    pub min_removed_blocks: usize,
    pub max_removed_blocks: usize,
    pub remove_pool_factor: usize,
    pub remove_count_sample_power: f64,
    pub remove_seed_per_block: (usize, usize),
    pub remove_entry_base_interval: usize,
    pub remove_entry_seed_candidate_count: usize,
    pub remove_seed_strategy_weights: RemoveSeedStrategyWeights,
    pub remove_seed_method_weights: RemoveSeedMethodWeights,
    pub remove_fluidity_slack_weight_range: (f64, f64),
    pub remove_fluidity_pref_spread_weight_range: (f64, f64),
    pub remove_x_distance_weight_max: f64,
    pub remove_y_distance_weight_max: f64,
    pub reconstruct_workload_weight_range: (f64, f64),
    pub reconstruct_volume_weight_range: (f64, f64),
    pub reconstruct_pref_spread_weight_range: (f64, f64),
    pub reconstruct_limit_time_urgency_weight_range: (f64, f64),
    pub reconstruct_order_random_weight_range: (f64, f64),
    pub shift_dy_range: (i64, i64),
    pub rotate_dy_range: (i64, i64),
    pub swap_neighbor_top_k: usize,
    pub swap_dy_range: (i64, i64),
    pub move_sample_blocks: usize,
    pub move_small_pool_size: usize,
}

impl NeighborParams {
    pub(crate) fn probabilities(&self) -> [(NeighborKind, f64); 5] {
        self.probabilities.weighted()
    }

    fn with_override(&self, value: &NeighborParamsOverride) -> Self {
        Self {
            probabilities: self.probabilities.with_override(&value.probabilities),
            min_removed_blocks: value.min_removed_blocks.unwrap_or(self.min_removed_blocks),
            max_removed_blocks: value.max_removed_blocks.unwrap_or(self.max_removed_blocks),
            remove_pool_factor: value.remove_pool_factor.unwrap_or(self.remove_pool_factor),
            remove_count_sample_power: value
                .remove_count_sample_power
                .unwrap_or(self.remove_count_sample_power),
            remove_seed_per_block: value
                .remove_seed_per_block
                .unwrap_or(self.remove_seed_per_block),
            remove_entry_base_interval: value
                .remove_entry_base_interval
                .unwrap_or(self.remove_entry_base_interval),
            remove_entry_seed_candidate_count: value
                .remove_entry_seed_candidate_count
                .unwrap_or(self.remove_entry_seed_candidate_count),
            remove_seed_strategy_weights: value
                .remove_seed_strategy_weights
                .unwrap_or(self.remove_seed_strategy_weights),
            remove_seed_method_weights: value
                .remove_seed_method_weights
                .unwrap_or(self.remove_seed_method_weights),
            remove_fluidity_slack_weight_range: value
                .remove_fluidity_slack_weight_range
                .unwrap_or(self.remove_fluidity_slack_weight_range),
            remove_fluidity_pref_spread_weight_range: value
                .remove_fluidity_pref_spread_weight_range
                .unwrap_or(self.remove_fluidity_pref_spread_weight_range),
            remove_x_distance_weight_max: value
                .remove_x_distance_weight_max
                .unwrap_or(self.remove_x_distance_weight_max),
            remove_y_distance_weight_max: value
                .remove_y_distance_weight_max
                .unwrap_or(self.remove_y_distance_weight_max),
            reconstruct_workload_weight_range: value
                .reconstruct_workload_weight_range
                .unwrap_or(self.reconstruct_workload_weight_range),
            reconstruct_volume_weight_range: value
                .reconstruct_volume_weight_range
                .unwrap_or(self.reconstruct_volume_weight_range),
            reconstruct_pref_spread_weight_range: value
                .reconstruct_pref_spread_weight_range
                .unwrap_or(self.reconstruct_pref_spread_weight_range),
            reconstruct_limit_time_urgency_weight_range: value
                .reconstruct_limit_time_urgency_weight_range
                .unwrap_or(self.reconstruct_limit_time_urgency_weight_range),
            reconstruct_order_random_weight_range: value
                .reconstruct_order_random_weight_range
                .unwrap_or(self.reconstruct_order_random_weight_range),
            shift_dy_range: value.shift_dy_range.unwrap_or(self.shift_dy_range),
            rotate_dy_range: value.rotate_dy_range.unwrap_or(self.rotate_dy_range),
            swap_neighbor_top_k: value
                .swap_neighbor_top_k
                .unwrap_or(self.swap_neighbor_top_k),
            swap_dy_range: value.swap_dy_range.unwrap_or(self.swap_dy_range),
            move_sample_blocks: value.move_sample_blocks.unwrap_or(self.move_sample_blocks),
            move_small_pool_size: value
                .move_small_pool_size
                .unwrap_or(self.move_small_pool_size),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoveSeedStrategyWeights {
    pub local_proximity: f64,
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
    fn weighted(&self) -> [(NeighborKind, f64); 5] {
        [
            (NeighborKind::LargeReconstruct, self.large_reconstruct),
            (NeighborKind::Shift, self.shift),
            (NeighborKind::Move, self.move_block),
            (NeighborKind::Rotate, self.rotate),
            (NeighborKind::Swap, self.swap),
        ]
    }

    fn with_override(&self, value: &NeighborProbabilitiesOverride) -> Self {
        Self {
            large_reconstruct: value.large_reconstruct.unwrap_or(self.large_reconstruct),
            shift: value.shift.unwrap_or(self.shift),
            move_block: value.move_block.unwrap_or(self.move_block),
            rotate: value.rotate.unwrap_or(self.rotate),
            swap: value.swap.unwrap_or(self.swap),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeighborParamsOverride {
    pub probabilities: NeighborProbabilitiesOverride,
    pub min_removed_blocks: Option<usize>,
    pub max_removed_blocks: Option<usize>,
    pub remove_pool_factor: Option<usize>,
    pub remove_count_sample_power: Option<f64>,
    pub remove_seed_per_block: Option<(usize, usize)>,
    pub remove_entry_base_interval: Option<usize>,
    pub remove_entry_seed_candidate_count: Option<usize>,
    pub remove_seed_strategy_weights: Option<RemoveSeedStrategyWeights>,
    pub remove_seed_method_weights: Option<RemoveSeedMethodWeights>,
    pub remove_fluidity_slack_weight_range: Option<(f64, f64)>,
    pub remove_fluidity_pref_spread_weight_range: Option<(f64, f64)>,
    pub remove_x_distance_weight_max: Option<f64>,
    pub remove_y_distance_weight_max: Option<f64>,
    pub reconstruct_workload_weight_range: Option<(f64, f64)>,
    pub reconstruct_volume_weight_range: Option<(f64, f64)>,
    pub reconstruct_pref_spread_weight_range: Option<(f64, f64)>,
    pub reconstruct_limit_time_urgency_weight_range: Option<(f64, f64)>,
    pub reconstruct_order_random_weight_range: Option<(f64, f64)>,
    pub shift_dy_range: Option<(i64, i64)>,
    pub rotate_dy_range: Option<(i64, i64)>,
    pub swap_neighbor_top_k: Option<usize>,
    pub swap_dy_range: Option<(i64, i64)>,
    pub move_sample_blocks: Option<usize>,
    pub move_small_pool_size: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NeighborProbabilitiesOverride {
    pub large_reconstruct: Option<f64>,
    pub shift: Option<f64>,
    #[serde(rename = "move")]
    pub move_block: Option<f64>,
    pub rotate: Option<f64>,
    pub swap: Option<f64>,
}
