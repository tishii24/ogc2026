use std::cmp::Reverse;

use crate::{
    Problem, ScheduledBlock,
    params::{InsertParams, ReconstructNeighborParams},
    utils::random::{Random, sample_weighted_index},
};

#[cfg(feature = "profile-reconstruct-weight")]
use super::reconstruct_weight_profile::{self, Outcome};

use super::{
    insert::{insert_greedy, sample_insert_anchor},
    objective::score13_block,
    precompute::Precompute,
};

#[derive(Clone, Copy)]
enum RemoveSeedMethod {
    Badness,
    Fluidity,
    Random,
}

#[derive(Clone, Copy)]
struct RemoveSeed {
    block_id: usize,
    remove_count: usize,
}

#[derive(Clone, Copy)]
pub(super) struct BlockOrderWeights {
    volume: f64,
    pref_spread: f64,
    limit_time_urgency: f64,
    slack_tightness: f64,
    random: f64,
}

impl BlockOrderWeights {
    fn normalize(&mut self) {
        let sum = self.volume
            + self.pref_spread
            + self.limit_time_urgency
            + self.slack_tightness
            + self.random;
        self.volume /= sum;
        self.pref_spread /= sum;
        self.limit_time_urgency /= sum;
        self.slack_tightness /= sum;
        self.random /= sum;
    }

    fn normalize_with_current_penalty(&mut self, current_penalty: &mut f64) -> [f64; 6] {
        let sum = self.volume
            + self.pref_spread
            + self.limit_time_urgency
            + self.slack_tightness
            + self.random
            + *current_penalty;
        self.volume /= sum;
        self.pref_spread /= sum;
        self.limit_time_urgency /= sum;
        self.slack_tightness /= sum;
        *current_penalty /= sum;
        self.random /= sum;
        [
            self.volume,
            self.pref_spread,
            self.limit_time_urgency,
            self.slack_tightness,
            *current_penalty,
            self.random,
        ]
    }
}

#[cfg(feature = "profile-reconstruct-weight")]
struct WeightOutcomeGuard {
    enabled: bool,
    weights: [f64; 6],
    outcome: Outcome,
}

#[cfg(feature = "profile-reconstruct-weight")]
impl Drop for WeightOutcomeGuard {
    fn drop(&mut self) {
        if self.enabled {
            reconstruct_weight_profile::record(self.weights, self.outcome);
        }
    }
}

struct BlockOrderContext {
    max_volume: f64,
    max_pref_spread: f64,
    max_limit_time: i64,
    limit_time_span: f64,
    max_slack: i64,
    slack_span: f64,
}

pub(super) fn sample_reconstruct_order_weights(
    rng: &mut impl Random,
    params: &ReconstructNeighborParams,
) -> BlockOrderWeights {
    BlockOrderWeights {
        volume: rng.gen_range_f64(params.volume_weight_range.0, params.volume_weight_range.1),
        pref_spread: rng.gen_range_f64(
            params.pref_spread_weight_range.0,
            params.pref_spread_weight_range.1,
        ),
        limit_time_urgency: rng.gen_range_f64(
            params.limit_time_urgency_weight_range.0,
            params.limit_time_urgency_weight_range.1,
        ),
        slack_tightness: rng.gen_range_f64(
            params.slack_tightness_weight_range.0,
            params.slack_tightness_weight_range.1,
        ),
        random: rng.gen_range_f64(
            params.order_random_weight_range.0,
            params.order_random_weight_range.1,
        ),
    }
}

pub(super) struct ReconstructBase {
    pub(super) schedule: Vec<ScheduledBlock>,
    pub(super) loads: Vec<f64>,
    pub(super) weighted_z1_z3: f64,
}

pub(super) fn build_reconstruct_base(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    removed_ids: &[usize],
) -> ReconstructBase {
    let mut removed = vec![false; problem.blocks.len()];
    for &block_id in removed_ids {
        removed[block_id] = true;
    }

    let mut remaining = Vec::with_capacity(schedule.len());
    let mut loads = vec![0.0; problem.bays.len()];
    let mut weighted_z1_z3 = 0.0;
    for &scheduled in schedule {
        if removed[scheduled.block_id] {
            continue;
        }
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        weighted_z1_z3 += score13_block(problem, pre, scheduled);
        remaining.push(scheduled);
    }

    ReconstructBase {
        schedule: remaining,
        loads,
        weighted_z1_z3,
    }
}

pub(super) struct LargeReconstructResult {
    pub(super) schedule: Vec<ScheduledBlock>,
    #[cfg(feature = "anneal-visualizer")]
    pub(super) selected_block_ids: Vec<usize>,
}

pub(super) fn try_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    params: &ReconstructNeighborParams,
    insert_params: &InsertParams,
    w2: f64,
    collect_weight_stats: bool,
) -> Option<LargeReconstructResult> {
    let k = sample_removed_count(rng, params).min(schedule.len());

    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, w2, rng, params)?;
    if removed_ids.is_empty() {
        return None;
    }
    let mut weights = sample_reconstruct_order_weights(rng, params);
    let mut current_penalty_weight = rng.gen_range_f64(
        params.current_penalty_weight_range.0,
        params.current_penalty_weight_range.1,
    );
    let weight_vector = weights.normalize_with_current_penalty(&mut current_penalty_weight);
    #[cfg(feature = "profile-reconstruct-weight")]
    let mut weight_outcome = WeightOutcomeGuard {
        enabled: collect_weight_stats,
        weights: weight_vector,
        outcome: Outcome::Failed,
    };
    #[cfg(not(feature = "profile-reconstruct-weight"))]
    let _ = (collect_weight_stats, weight_vector);
    let mut current_penalties = vec![0.0; problem.blocks.len()];
    let mut original_by_id = vec![None; problem.blocks.len()];
    for &scheduled in schedule {
        current_penalties[scheduled.block_id] = score13_block(problem, pre, scheduled);
        original_by_id[scheduled.block_id] = Some(scheduled);
    }
    sort_block_order(
        problem,
        &pre.max_footprint_area,
        &pre.pref_spread,
        &mut removed_ids,
        weights,
        Some(&current_penalties),
        current_penalty_weight,
        rng,
    );

    let ReconstructBase {
        schedule: mut cur,
        mut loads,
        weighted_z1_z3: mut fixed_score13,
    } = build_reconstruct_base(problem, pre, schedule, &removed_ids);
    let anchor = sample_insert_anchor(rng, insert_params);
    #[cfg(feature = "profile-reconstruct-weight")]
    let mut changed = false;

    for &block_id in &removed_ids {
        if fixed_score13 > accept_threshold + 1e-9 {
            return None;
        }
        let old = original_by_id[block_id]?;
        let block = &problem.blocks[block_id];
        let max_tardiness =
            ((accept_threshold - fixed_score13) / problem.weights.w1.max(1e-4)).floor() as i64;
        let max_entry_time = block
            .due_date
            .saturating_add(max_tardiness)
            .saturating_sub(block.processing_time);
        let scheduled = insert_greedy(
            problem,
            pre,
            old,
            block.release_time,
            max_entry_time,
            &cur,
            &loads,
            insert_params,
            &pre.bay_order_by_pref[old.block_id],
            params.insert_candidate_top_k,
            params.insert_candidate_select_p,
            w2,
            anchor,
            rng,
        )?;
        #[cfg(feature = "profile-reconstruct-weight")]
        {
            changed |= scheduled != old;
        }
        loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        fixed_score13 += score13_block(problem, pre, scheduled);
        cur.push(scheduled);
    }

    #[cfg(feature = "profile-reconstruct-weight")]
    {
        weight_outcome.outcome = if changed {
            Outcome::Changed
        } else {
            Outcome::Unchanged
        };
    }
    Some(LargeReconstructResult {
        schedule: cur,
        #[cfg(feature = "anneal-visualizer")]
        selected_block_ids: removed_ids,
    })
}

pub(super) fn sample_removed_count<R: Random>(
    rng: &mut R,
    params: &ReconstructNeighborParams,
) -> usize {
    params.remove_count.sample(rng)
}

fn scheduled_center(pre: &Precompute, s: ScheduledBlock) -> (f64, f64) {
    let (cx, cy) = pre.orientation_bbox_center[s.block_id][s.orient_idx];
    (s.x as f64 + cx, s.y as f64 + cy)
}

fn push_removed_block(selected: &mut Vec<usize>, used: &mut [bool], block_id: usize, limit: usize) {
    if selected.len() < limit && !used[block_id] {
        selected.push(block_id);
        used[block_id] = true;
    }
}

fn remove_badness(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    w2: f64,
) -> Vec<i64> {
    let mut loads = vec![0.0; problem.bays.len()];
    for s in schedule {
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
    }
    let heavy_bay = loads
        .iter()
        .enumerate()
        .max_by(|&(a, a_load), &(b, b_load)| {
            (a_load * pre.bay_load_scale[a]).total_cmp(&(b_load * pre.bay_load_scale[b]))
        })
        .map(|(bay_id, _)| bay_id);

    let mut badness = vec![0i64; problem.blocks.len()];
    for s in schedule {
        let block = &problem.blocks[s.block_id];
        let tardiness = (s.exit_time - block.due_date).max(0);
        let pref_penalty = pre.pref_penalty[s.block_id][s.bay_id];
        let score = tardiness as f64 * problem.weights.w1
            + pref_penalty as f64 * problem.weights.w3
            + if Some(s.bay_id) == heavy_bay { w2 } else { 0.0 };
        badness[s.block_id] = score as i64;
    }
    badness
}

fn choose_local_proximity_seeds<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    by_block: &[Option<ScheduledBlock>],
    k: usize,
    bad_pool: &[usize],
    rng: &mut R,
    params: &ReconstructNeighborParams,
) -> Option<Vec<RemoveSeed>> {
    let pool_len = (k * params.remove_pool_factor).min(schedule.len()).max(k);
    let slack_weight = rng.gen_range_f64(
        params.remove_fluidity_slack_weight_range.0,
        params.remove_fluidity_slack_weight_range.1,
    );
    let pref_spread_weight = rng.gen_range_f64(
        params.remove_fluidity_pref_spread_weight_range.0,
        params.remove_fluidity_pref_spread_weight_range.1,
    );
    let max_slack = schedule
        .iter()
        .map(|scheduled| {
            let block = &problem.blocks[scheduled.block_id];
            (block.due_date - block.release_time - block.processing_time).max(0) as f64
        })
        .fold(0.0, f64::max)
        .max(1.0);
    let max_pref_spread = schedule
        .iter()
        .map(|scheduled| pre.pref_spread[scheduled.block_id] as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let mut fluidity = vec![0.0; problem.blocks.len()];
    for scheduled in schedule {
        let block = &problem.blocks[scheduled.block_id];
        let slack = (block.due_date - block.release_time - block.processing_time).max(0) as f64;
        let pref_spread = pre.pref_spread[scheduled.block_id] as f64;
        fluidity[scheduled.block_id] = slack_weight * slack / max_slack
            + pref_spread_weight * (1.0 - pref_spread / max_pref_spread);
    }
    let mut fluid_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut fluid_pool);
    fluid_pool.sort_by(|&a, &b| fluidity[b].total_cmp(&fluidity[a]));
    fluid_pool.truncate(pool_len);

    let mut seed_sizes = Vec::new();
    let mut seed_size_sum = 0;
    while seed_size_sum < k {
        let seed_size = params
            .remove_blocks_per_seed
            .sample(rng)
            .min(k - seed_size_sum);
        seed_sizes.push(seed_size);
        seed_size_sum += seed_size;
    }
    let seed_count = seed_sizes.len();
    let entry_base_interval =
        sample_weighted_index(rng, &params.remove_entry_base_interval_weights) + 1;
    let base_count = (1 + (seed_count - 1) / entry_base_interval).min(seed_count);
    let method_weights = params.remove_seed_method_weights;
    let total_method_weight =
        method_weights.badness + method_weights.fluidity + method_weights.random;
    let seed_methods: Vec<_> = (0..seed_count)
        .map(|_| {
            let value = rng.next_f64() * total_method_weight;
            if value < method_weights.badness {
                RemoveSeedMethod::Badness
            } else if value < method_weights.badness + method_weights.fluidity {
                RemoveSeedMethod::Fluidity
            } else {
                RemoveSeedMethod::Random
            }
        })
        .collect();

    let mut used = vec![false; problem.blocks.len()];
    let mut seeds = Vec::with_capacity(seed_count);
    let mut entry_bases = Vec::with_capacity(base_count);

    for seed_index in 0..base_count {
        let mut seed_pool: Vec<usize> = match seed_methods[seed_index] {
            RemoveSeedMethod::Badness => bad_pool.to_vec(),
            RemoveSeedMethod::Fluidity => fluid_pool.clone(),
            RemoveSeedMethod::Random => schedule.iter().map(|s| s.block_id).collect(),
        };
        seed_pool.retain(|&block_id| !used[block_id]);
        rng.shuffle(&mut seed_pool);
        let seed_id = *seed_pool.get(0)?;
        used[seed_id] = true;
        seeds.push(RemoveSeed {
            block_id: seed_id,
            remove_count: seed_sizes[seed_index],
        });
        entry_bases.push(seed_id);
    }

    for seed_index in base_count..seed_count {
        let base_id = entry_bases[rng.gen_range(0, entry_bases.len())];
        let base = by_block[base_id].unwrap();
        let mut seed_pool: Vec<usize> = match seed_methods[seed_index] {
            RemoveSeedMethod::Badness => bad_pool.to_vec(),
            RemoveSeedMethod::Fluidity => fluid_pool.clone(),
            RemoveSeedMethod::Random => schedule.iter().map(|s| s.block_id).collect(),
        };
        seed_pool.retain(|&block_id| !used[block_id]);
        rng.shuffle(&mut seed_pool);
        seed_pool.sort_by_key(|&block_id| {
            let candidate = by_block[block_id].unwrap();
            (
                candidate.entry_time.abs_diff(base.entry_time),
                candidate.bay_id == base.bay_id,
            )
        });
        seed_pool.truncate(
            params
                .remove_entry_seed_candidate_count
                .min(seed_pool.len()),
        );
        rng.shuffle(&mut seed_pool);
        let seed_id = *seed_pool.get(0)?;
        used[seed_id] = true;
        seeds.push(RemoveSeed {
            block_id: seed_id,
            remove_count: seed_sizes[seed_index],
        });
    }

    Some(seeds)
}

fn collect_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    by_block: &[Option<ScheduledBlock>],
    k: usize,
    seeds: &[RemoveSeed],
    bad_pool: &[usize],
    remove_x_distance_weight: f64,
    remove_y_distance_weight: f64,
    remove_t_distance_weight: f64,
    remove_distance_power: f64,
    rng: &mut R,
) -> Vec<usize> {
    let mut selected = Vec::with_capacity(k);
    let mut used = vec![false; problem.blocks.len()];
    for seed in seeds {
        push_removed_block(&mut selected, &mut used, seed.block_id, k);
    }

    for seed in seeds {
        let scheduled = by_block[seed.block_id].unwrap();
        let (sx, sy) = scheduled_center(pre, scheduled);
        let st = scheduled.entry_time as f64;
        let mut neighbors: Vec<(f64, usize)> = schedule
            .iter()
            .filter(|s| s.bay_id == scheduled.bay_id && !used[s.block_id])
            .map(|&candidate| {
                let (x, y) = scheduled_center(pre, candidate);
                let dx = x - sx;
                let dy = y - sy;
                let dt = candidate.entry_time as f64 - st;
                (
                    remove_x_distance_weight * dx.abs().powf(remove_distance_power)
                        + remove_y_distance_weight * dy.abs().powf(remove_distance_power)
                        + remove_t_distance_weight * dt.abs().powf(remove_distance_power),
                    candidate.block_id,
                )
            })
            .collect();
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, block_id) in neighbors.into_iter().take(seed.remove_count - 1) {
            push_removed_block(&mut selected, &mut used, block_id, k);
        }
    }

    let mut fill_pool = bad_pool.to_vec();
    rng.shuffle(&mut fill_pool);
    for block_id in fill_pool {
        push_removed_block(&mut selected, &mut used, block_id, k);
    }
    selected
}

pub(super) fn choose_removed_blocks<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    k: usize,
    w2: f64,
    rng: &mut R,
    params: &ReconstructNeighborParams,
) -> Option<Vec<usize>> {
    if schedule.is_empty() || k == 0 {
        return None;
    }

    let k = k.min(schedule.len());
    let remove_x_distance_weight = rng.gen_range_f64(
        params.remove_x_distance_weight_range.0,
        params.remove_x_distance_weight_range.1,
    );
    let remove_y_distance_weight = rng.gen_range_f64(
        params.remove_y_distance_weight_range.0,
        params.remove_y_distance_weight_range.1,
    );
    let remove_t_distance_weight = rng.gen_range_f64(
        params.remove_t_distance_weight_range.0,
        params.remove_t_distance_weight_range.1,
    );
    let remove_distance_power = rng.gen_range_f64(
        params.remove_distance_power_range.0,
        params.remove_distance_power_range.1,
    );
    let badness = remove_badness(problem, pre, schedule, w2);
    let mut by_block = vec![None; problem.blocks.len()];
    for &scheduled in schedule {
        by_block[scheduled.block_id] = Some(scheduled);
    }

    let mut bad_pool: Vec<usize> = schedule.iter().map(|s| s.block_id).collect();
    rng.shuffle(&mut bad_pool);
    bad_pool.sort_by_key(|&block_id| Reverse(badness[block_id]));
    let pool_len = (k * params.remove_pool_factor).min(schedule.len()).max(k);
    bad_pool.truncate(pool_len);

    let seeds =
        choose_local_proximity_seeds(problem, pre, schedule, &by_block, k, &bad_pool, rng, params)?;
    let blocks = collect_removed_blocks(
        problem,
        pre,
        schedule,
        &by_block,
        k,
        &seeds,
        &bad_pool,
        remove_x_distance_weight,
        remove_y_distance_weight,
        remove_t_distance_weight,
        remove_distance_power,
        rng,
    );
    Some(blocks)
}

fn block_volume(problem: &Problem, block_areas: &[f64], block_id: usize) -> f64 {
    block_areas[block_id] * problem.blocks[block_id].processing_time as f64
}

fn block_limit_time(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    block.due_date - block.processing_time
}

fn block_slack(problem: &Problem, block_id: usize) -> i64 {
    let block = &problem.blocks[block_id];
    (block.due_date - block.processing_time - block.release_time).max(0)
}

fn build_block_order_context(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &[usize],
) -> BlockOrderContext {
    let max_volume = order
        .iter()
        .map(|&block_id| block_volume(problem, block_areas, block_id))
        .fold(0.0, f64::max)
        .max(1.0);
    let max_pref_spread = order
        .iter()
        .map(|&block_id| pref_spread[block_id] as f64)
        .fold(0.0, f64::max)
        .max(1.0);
    let min_limit_time = order
        .iter()
        .map(|&block_id| block_limit_time(problem, block_id))
        .min()
        .unwrap_or(0);
    let max_limit_time = order
        .iter()
        .map(|&block_id| block_limit_time(problem, block_id))
        .max()
        .unwrap_or(min_limit_time);
    let min_slack = order
        .iter()
        .map(|&block_id| block_slack(problem, block_id))
        .min()
        .unwrap_or(0);
    let max_slack = order
        .iter()
        .map(|&block_id| block_slack(problem, block_id))
        .max()
        .unwrap_or(min_slack);

    BlockOrderContext {
        max_volume,
        max_pref_spread,
        max_limit_time,
        limit_time_span: (max_limit_time - min_limit_time).max(1) as f64,
        max_slack,
        slack_span: (max_slack - min_slack).max(1) as f64,
    }
}

fn block_order_score(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    ctx: &BlockOrderContext,
    weights: BlockOrderWeights,
    current_penalties: Option<&[f64]>,
    current_penalty_weight: f64,
    max_current_penalty: f64,
    block_id: usize,
) -> f64 {
    let volume_norm = block_volume(problem, block_areas, block_id) / ctx.max_volume;
    let pref_spread_norm = pref_spread[block_id] as f64 / ctx.max_pref_spread;
    let limit_time_urgency =
        (ctx.max_limit_time - block_limit_time(problem, block_id)) as f64 / ctx.limit_time_span;
    let slack_tightness = (ctx.max_slack - block_slack(problem, block_id)) as f64 / ctx.slack_span;
    let current_penalty = current_penalties
        .map(|penalties| penalties[block_id] / max_current_penalty)
        .unwrap_or(0.0);

    weights.volume * volume_norm
        + weights.pref_spread * pref_spread_norm
        + weights.limit_time_urgency * limit_time_urgency
        + weights.slack_tightness * slack_tightness
        + current_penalty_weight * current_penalty
}

pub(super) fn sort_block_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &mut [usize],
    weights: BlockOrderWeights,
    current_penalties: Option<&[f64]>,
    current_penalty_weight: f64,
    rng: &mut R,
) {
    let ctx = build_block_order_context(problem, block_areas, pref_spread, order);
    let max_current_penalty = current_penalties
        .map(|penalties| {
            order
                .iter()
                .map(|&block_id| penalties[block_id])
                .fold(0.0, f64::max)
                .max(1.0)
        })
        .unwrap_or(1.0);
    let mut random_scores = vec![0.0; problem.blocks.len()];
    for &block_id in order.iter() {
        random_scores[block_id] = rng.gen_range_f64(0., weights.random);
    }
    order.sort_by(|&a, &b| {
        let score_a = block_order_score(
            problem,
            block_areas,
            pref_spread,
            &ctx,
            weights,
            current_penalties,
            current_penalty_weight,
            max_current_penalty,
            a,
        ) + random_scores[a];
        let score_b = block_order_score(
            problem,
            block_areas,
            pref_spread,
            &ctx,
            weights,
            current_penalties,
            current_penalty_weight,
            max_current_penalty,
            b,
        ) + random_scores[b];
        score_b
            .total_cmp(&score_a)
            .then(block_limit_time(problem, a).cmp(&block_limit_time(problem, b)))
            .then(
                block_volume(problem, block_areas, b).total_cmp(&block_volume(
                    problem,
                    block_areas,
                    a,
                )),
            )
            .then(problem.blocks[b].workload.cmp(&problem.blocks[a].workload))
            .then(pref_spread[b].cmp(&pref_spread[a]))
            .then(a.cmp(&b))
    });
}

pub(super) fn sort_default_reconstruct_order<R: Random>(
    problem: &Problem,
    block_areas: &[f64],
    pref_spread: &[i64],
    order: &mut [usize],
    rng: &mut R,
    params: &ReconstructNeighborParams,
) {
    let mut weights = sample_reconstruct_order_weights(rng, params);
    weights.normalize();
    sort_block_order(
        problem,
        block_areas,
        pref_spread,
        order,
        weights,
        None,
        0.0,
        rng,
    );
}
