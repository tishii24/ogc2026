use crate::{Problem, ScheduledBlock, params::ReconstructNeighborParams, utils::random::Random};

use super::{
    objective::{ScheduleScore, normalized_imbalance, score_schedule, score13_block},
    placement_scan::{Interval, PlacementXScanner, XRange, find_leftmost_fixed_time_x},
    precompute::Precompute,
    reconstruct::{
        EntryTimeBounds, HeuristicPrecedence, ReconstructBase, build_reconstruct_base,
        build_topological_order, choose_removed_blocks, precedence_entry_time_bounds,
        sample_reconstruct_order_weights, sample_removed_count, scheduled_by_id, sort_block_order,
    },
};

const HASH_OFFSET: u64 = 1469598103934665603;

struct BaseInsertTemplate {
    bay_id: usize,
    orient_idx: usize,
    y: i64,
    x_ranges: Vec<XRange>,
    base_forbidden_intervals: Vec<Interval>,
    sort_entry_time: i64,
    sort_score13: f64,
}

#[derive(Clone)]
struct BeamState {
    added: Vec<ScheduledBlock>,
    projected_loads: Vec<f64>,
    score13: f64,
    score: f64,
    exact_hash: u64,
    placement_hash: u64,
    changed: bool,
}

#[derive(Clone, Copy)]
struct PlacementCandidate {
    scheduled: ScheduledBlock,
    score13: f64,
    score: f64,
    exact_hash: u64,
    placement_hash: u64,
    changed: bool,
}

pub(super) fn try_beam_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    constraints: Option<&HeuristicPrecedence>,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    params: &ReconstructNeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng, params).min(problem.blocks.len());
    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng, params)?;
    if removed_ids.is_empty() {
        return None;
    }

    let weights = sample_reconstruct_order_weights(rng, params);
    let order = if let Some(constraints) = constraints {
        build_topological_order(
            problem,
            pre,
            &removed_ids,
            constraints,
            weights,
            None,
            0.0,
            rng,
        )
    } else {
        sort_block_order(
            problem,
            &pre.max_footprint_area,
            &pre.pref_spread,
            &mut removed_ids,
            weights,
            None,
            0.0,
            rng,
        );
        removed_ids.clone()
    };

    let original_by_id = scheduled_by_id(problem, schedule);
    let ReconstructBase {
        schedule: base,
        loads: _,
        weighted_z1_z3: base_score13,
    } = build_reconstruct_base(problem, pre, schedule, &removed_ids);
    if base_score13 > accept_threshold + 1e-9 {
        return None;
    }

    let mut templates_by_id: Vec<Vec<BaseInsertTemplate>> =
        (0..problem.blocks.len()).map(|_| Vec::new()).collect();
    for &block_id in &order {
        templates_by_id[block_id] = build_base_templates(
            problem,
            pre,
            &base,
            block_id,
            params.beam.candidate_pool_count,
            params.beam.candidate_group_limit,
        );
    }

    let base_by_id = scheduled_by_id(problem, &base);
    let mut projected_loads = vec![0.0; problem.bays.len()];
    for scheduled in schedule {
        projected_loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
    }
    let initial_score = base_score13
        + problem.weights.w2 * normalized_imbalance(&projected_loads, &pre.bay_load_scale);
    let mut beam = vec![BeamState {
        added: Vec::with_capacity(order.len()),
        projected_loads,
        score13: base_score13,
        score: initial_score,
        exact_hash: 0,
        placement_hash: 0,
        changed: false,
    }];

    for (depth, &block_id) in order.iter().enumerate() {
        let original = original_by_id[block_id]?;
        let block = &problem.blocks[block_id];
        let mut next = Vec::with_capacity(beam.len() * params.beam.candidate_count.max(1));

        for state in &beam {
            let Some(bounds) =
                entry_time_bounds_for_state(problem, constraints, &base_by_id, state, block_id)
            else {
                continue;
            };
            let mut placements = Vec::with_capacity(params.beam.candidate_count);

            for template in &templates_by_id[block_id] {
                let Some(scheduled) =
                    materialize_template(problem, pre, state, block_id, template, bounds)
                else {
                    continue;
                };
                push_placement_candidate(problem, pre, state, original, scheduled, &mut placements);
                if placements.len() >= params.beam.candidate_count {
                    break;
                }
            }
            retain_parent_candidates(&mut placements, params.beam.candidate_count);

            for placement in placements {
                if placement.score13 > accept_threshold + 1e-9 {
                    continue;
                }
                let mut added = state.added.clone();
                added.push(placement.scheduled);
                let mut projected_loads = state.projected_loads.clone();
                let workload = block.workload as f64;
                projected_loads[original.bay_id] -= workload;
                projected_loads[placement.scheduled.bay_id] += workload;
                next.push(BeamState {
                    added,
                    projected_loads,
                    score13: placement.score13,
                    score: placement.score,
                    exact_hash: placement.exact_hash,
                    placement_hash: placement.placement_hash,
                    changed: placement.changed,
                });
            }
        }

        let next_block_id = order.get(depth + 1).copied();
        beam = select_beam(
            next,
            params.beam.width,
            params.beam.state_group_limit,
            |state| {
                next_block_id.is_none_or(|next_block_id| {
                    state_has_placement(
                        problem,
                        pre,
                        constraints,
                        &base_by_id,
                        state,
                        next_block_id,
                        &templates_by_id[next_block_id],
                        accept_threshold,
                    )
                })
            },
        );
        if beam.is_empty() {
            return None;
        }
    }

    let mut best: Option<(f64, Vec<ScheduledBlock>)> = None;
    for state in beam {
        if !state.changed {
            continue;
        }
        let mut blocks = base.clone();
        blocks.extend(state.added);
        let ScheduleScore {
            objective: score, ..
        } = score_schedule(problem, pre, &blocks);
        if score > accept_threshold + 1e-9 {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            best = Some((score, blocks));
        }
    }
    best.map(|(_, blocks)| blocks)
}

fn build_base_templates(
    problem: &Problem,
    pre: &Precompute,
    base: &[ScheduledBlock],
    block_id: usize,
    candidate_pool_count: usize,
    candidate_group_limit: usize,
) -> Vec<BaseInsertTemplate> {
    if candidate_pool_count == 0 || candidate_group_limit == 0 {
        return Vec::new();
    }

    let block = &problem.blocks[block_id];
    let mut groups: Vec<Vec<Vec<BaseInsertTemplate>>> = (0..problem.bays.len())
        .map(|_| (0..block.shape.len()).map(|_| Vec::new()).collect())
        .collect();

    for bay_id in 0..problem.bays.len() {
        let Some(mut scanner) = PlacementXScanner::new(
            problem,
            pre,
            base,
            block_id,
            bay_id,
            block.release_time,
            i64::MAX,
        ) else {
            continue;
        };
        for orient_idx in 0..block.shape.len() {
            let Some(fit_range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            let group = &mut groups[bay_id][orient_idx];
            for y in fit_range.min_y..=fit_range.max_y {
                scanner.scan_y_ranges(
                    orient_idx,
                    y,
                    fit_range.min_x,
                    fit_range.max_x,
                    |range, forbidden| {
                        let Some(sort_entry_time) =
                            forbidden.first_feasible_time(&[], block.release_time, i64::MAX)
                        else {
                            return false;
                        };
                        let mut base_forbidden_intervals = Vec::new();
                        forbidden.collect_merged(&mut base_forbidden_intervals);
                        if let Some(template) = group.iter_mut().find(|template| {
                            template.y == y
                                && template.base_forbidden_intervals == base_forbidden_intervals
                        }) {
                            template.x_ranges.push(range);
                            return false;
                        }

                        let scheduled = ScheduledBlock {
                            block_id,
                            bay_id,
                            orient_idx,
                            x: range.min_x,
                            y,
                            entry_time: sort_entry_time,
                            exit_time: sort_entry_time + block.processing_time,
                        };
                        group.push(BaseInsertTemplate {
                            bay_id,
                            orient_idx,
                            y,
                            x_ranges: vec![range],
                            base_forbidden_intervals,
                            sort_entry_time,
                            sort_score13: score13_block(problem, pre, scheduled),
                        });
                        false
                    },
                );
                for template in group.iter_mut() {
                    merge_x_ranges(&mut template.x_ranges);
                }
                group.sort_by(base_template_cmp);
                group.truncate(candidate_group_limit);
            }
        }
    }

    let mut candidate_groups = Vec::new();
    for bay_groups in groups {
        for mut group in bay_groups {
            if group.is_empty() {
                continue;
            }
            group.sort_by(base_template_cmp);
            group.truncate(candidate_group_limit);
            candidate_groups.push(group);
        }
    }
    candidate_groups.sort_by(|a, b| base_template_cmp(&a[0], &b[0]));

    let mut iterators: Vec<_> = candidate_groups
        .into_iter()
        .map(|group| group.into_iter())
        .collect();
    let mut result = Vec::with_capacity(candidate_pool_count);
    loop {
        let mut progressed = false;
        for iter in &mut iterators {
            if let Some(template) = iter.next() {
                result.push(template);
                progressed = true;
                if result.len() >= candidate_pool_count {
                    return result;
                }
            }
        }
        if !progressed {
            return result;
        }
    }
}

fn merge_x_ranges(ranges: &mut Vec<XRange>) {
    ranges.sort_by_key(|range| (range.min_x, range.max_x));
    let mut merged: Vec<XRange> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        if let Some(last) = merged.last_mut()
            && range.min_x <= last.max_x.saturating_add(1)
        {
            last.max_x = last.max_x.max(range.max_x);
        } else {
            merged.push(range);
        }
    }
    *ranges = merged;
}

fn base_template_cmp(a: &BaseInsertTemplate, b: &BaseInsertTemplate) -> std::cmp::Ordering {
    a.sort_score13
        .total_cmp(&b.sort_score13)
        .then(a.sort_entry_time.cmp(&b.sort_entry_time))
        .then(a.bay_id.cmp(&b.bay_id))
        .then(a.orient_idx.cmp(&b.orient_idx))
        .then(a.y.cmp(&b.y))
        .then(
            a.x_ranges
                .first()
                .unwrap()
                .min_x
                .cmp(&b.x_ranges.first().unwrap().min_x),
        )
}

fn entry_time_bounds_for_state(
    problem: &Problem,
    constraints: Option<&HeuristicPrecedence>,
    base_by_id: &[Option<ScheduledBlock>],
    state: &BeamState,
    block_id: usize,
) -> Option<EntryTimeBounds> {
    if let Some(constraints) = constraints {
        let mut current_by_id = base_by_id.to_vec();
        for &scheduled in &state.added {
            current_by_id[scheduled.block_id] = Some(scheduled);
        }
        precedence_entry_time_bounds(problem, constraints, &current_by_id, block_id)
    } else {
        Some(EntryTimeBounds {
            min: problem.blocks[block_id].release_time,
            max: i64::MAX,
        })
    }
}

fn materialize_template(
    problem: &Problem,
    pre: &Precompute,
    state: &BeamState,
    block_id: usize,
    template: &BaseInsertTemplate,
    bounds: EntryTimeBounds,
) -> Option<ScheduledBlock> {
    let block = &problem.blocks[block_id];
    let entry_time = first_feasible_time(
        &template.base_forbidden_intervals,
        bounds.min.max(block.release_time),
        bounds.max,
    )?;
    for &range in &template.x_ranges {
        let Some(x) = find_leftmost_fixed_time_x(
            problem,
            pre,
            block_id,
            template.bay_id,
            template.orient_idx,
            template.y,
            entry_time,
            range,
            &state.added,
        ) else {
            continue;
        };
        return Some(ScheduledBlock {
            block_id,
            bay_id: template.bay_id,
            orient_idx: template.orient_idx,
            x,
            y: template.y,
            entry_time,
            exit_time: entry_time + block.processing_time,
        });
    }
    None
}

fn first_feasible_time(intervals: &[Interval], min_t: i64, max_t: i64) -> Option<i64> {
    let mut t = min_t;
    let mut pos = intervals.partition_point(|&(_, right)| right < t);
    while let Some(&(left, right)) = intervals.get(pos) {
        if left > t {
            break;
        }
        t = right.checked_add(1)?;
        if t > max_t {
            return None;
        }
        pos += 1;
    }
    (t <= max_t).then_some(t)
}

fn state_has_placement(
    problem: &Problem,
    pre: &Precompute,
    constraints: Option<&HeuristicPrecedence>,
    base_by_id: &[Option<ScheduledBlock>],
    state: &BeamState,
    block_id: usize,
    templates: &[BaseInsertTemplate],
    accept_threshold: f64,
) -> bool {
    let Some(bounds) =
        entry_time_bounds_for_state(problem, constraints, base_by_id, state, block_id)
    else {
        return false;
    };
    templates.iter().any(|template| {
        materialize_template(problem, pre, state, block_id, template, bounds).is_some_and(
            |scheduled| {
                state.score13 + score13_block(problem, pre, scheduled) <= accept_threshold + 1e-9
            },
        )
    })
}

fn push_placement_candidate(
    problem: &Problem,
    pre: &Precompute,
    state: &BeamState,
    original: ScheduledBlock,
    scheduled: ScheduledBlock,
    placements: &mut Vec<PlacementCandidate>,
) {
    let block = &problem.blocks[scheduled.block_id];
    let score13 = state.score13 + score13_block(problem, pre, scheduled);
    let imbalance = normalized_imbalance_with_replace(
        pre,
        &state.projected_loads,
        original.bay_id,
        scheduled.bay_id,
        block.workload as f64,
    );
    placements.push(PlacementCandidate {
        scheduled,
        score13,
        score: score13 + problem.weights.w2 * imbalance,
        exact_hash: state.exact_hash ^ hash_scheduled_block(scheduled),
        placement_hash: state.placement_hash ^ hash_block_placement(scheduled),
        changed: state.changed || scheduled != original,
    });
}

fn normalized_imbalance_with_replace(
    pre: &Precompute,
    loads: &[f64],
    old_bay: usize,
    new_bay: usize,
    workload: f64,
) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let mut load = load;
        if bay_id == old_bay {
            load -= workload;
        }
        if bay_id == new_bay {
            load += workload;
        }
        let normalized = load * pre.bay_load_scale[bay_id];
        min_value = min_value.min(normalized);
        max_value = max_value.max(normalized);
    }
    (max_value - min_value).floor()
}

fn retain_parent_candidates(candidates: &mut Vec<PlacementCandidate>, limit: usize) {
    candidates.sort_by(placement_candidate_cmp);
    candidates.dedup_by(|a, b| a.scheduled == b.scheduled);
    candidates.truncate(limit);
}

fn placement_candidate_cmp(a: &PlacementCandidate, b: &PlacementCandidate) -> std::cmp::Ordering {
    a.score
        .total_cmp(&b.score)
        .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
        .then(a.scheduled.bay_id.cmp(&b.scheduled.bay_id))
        .then(a.scheduled.orient_idx.cmp(&b.scheduled.orient_idx))
        .then(a.scheduled.y.cmp(&b.scheduled.y))
        .then(a.scheduled.x.cmp(&b.scheduled.x))
}

fn select_beam(
    mut candidates: Vec<BeamState>,
    width: usize,
    group_limit: usize,
    mut is_viable: impl FnMut(&BeamState) -> bool,
) -> Vec<BeamState> {
    if width == 0 {
        return Vec::new();
    }
    candidates.sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then(a.exact_hash.cmp(&b.exact_hash))
    });

    let mut selected = Vec::with_capacity(width);
    let mut deferred = Vec::new();
    for candidate in candidates {
        if contains_exact(&selected, &candidate) {
            continue;
        }
        let group_count = selected
            .iter()
            .filter(|state| state.placement_hash == candidate.placement_hash)
            .count();
        if group_count >= group_limit {
            deferred.push(candidate);
            continue;
        }
        if is_viable(&candidate) {
            selected.push(candidate);
            if selected.len() >= width {
                break;
            }
        }
    }
    if selected.len() < width {
        for candidate in deferred {
            if contains_exact(&selected, &candidate) || !is_viable(&candidate) {
                continue;
            }
            selected.push(candidate);
            if selected.len() >= width {
                break;
            }
        }
    }
    selected.sort_by(|a, b| a.score.total_cmp(&b.score));
    selected
}

fn contains_exact(states: &[BeamState], candidate: &BeamState) -> bool {
    states
        .iter()
        .any(|state| state.exact_hash == candidate.exact_hash && state.added == candidate.added)
}

fn hash_block_placement(block: ScheduledBlock) -> u64 {
    let mut hash = HASH_OFFSET;
    hash = mix_hash(hash, block.block_id as u64);
    hash = mix_hash(hash, block.bay_id as u64);
    mix_hash(hash, block.orient_idx as u64)
}

fn hash_scheduled_block(block: ScheduledBlock) -> u64 {
    let mut hash = hash_block_placement(block);
    hash = mix_hash(hash, block.x as u64);
    hash = mix_hash(hash, block.y as u64);
    mix_hash(hash, block.entry_time as u64)
}

fn mix_hash(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash.wrapping_mul(1099511628211)
}
