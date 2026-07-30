use crate::{Problem, ScheduledBlock, params::ReconstructNeighborParams, utils::random::Random};

use super::{
    objective::{ScheduleScore, score_schedule, score13_block},
    placement_scan::{Interval, PlacementXScanner, XRange, find_leftmost_fixed_time_x},
    precompute::Precompute,
    reconstruct::{
        EntryTimeBounds, HeuristicPrecedence, ReconstructBase, build_reconstruct_base,
        build_topological_order, choose_removed_blocks, sample_reconstruct_order_weights,
        sample_removed_count, scheduled_by_id, sort_block_order,
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
    exact_hash: u64,
    placement_hash: u64,
    changed: bool,
}

struct BeamPrecedenceContext {
    base_bounds: Vec<EntryTimeBounds>,
    removed_predecessor_positions: Vec<Vec<usize>>,
}

#[derive(Clone, Copy)]
struct BeamCandidate {
    parent_index: usize,
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

    let mut templates_by_id: Vec<Option<Vec<BaseInsertTemplate>>> =
        (0..problem.blocks.len()).map(|_| None).collect();

    let base_by_id = scheduled_by_id(problem, &base);
    let precedence_context = if let Some(constraints) = constraints {
        Some(build_beam_precedence_context(
            problem,
            constraints,
            &base_by_id,
            &order,
        )?)
    } else {
        None
    };
    let mut projected_loads = vec![0.0; problem.bays.len()];
    for scheduled in schedule {
        projected_loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
    }
    let mut beam = vec![BeamState {
        added: Vec::with_capacity(order.len()),
        projected_loads,
        score13: base_score13,
        exact_hash: 0,
        placement_hash: 0,
        changed: false,
    }];

    for &block_id in &order {
        let original = original_by_id[block_id]?;
        if templates_by_id[block_id].is_none() {
            let template_bounds = precedence_context.as_ref().map_or(
                EntryTimeBounds {
                    min: problem.blocks[block_id].release_time,
                    max: i64::MAX,
                },
                |context| context.base_bounds[block_id],
            );
            templates_by_id[block_id] = Some(build_base_templates(
                problem,
                pre,
                &base,
                block_id,
                original.orient_idx,
                template_bounds,
                params.beam.candidate_pool_count,
                params.beam.candidate_group_limit,
                params.beam.orientation_sample_count,
                params.beam.y_sample_count,
                rng,
            ));
        }
        let templates = templates_by_id[block_id].as_ref().unwrap();
        let block = &problem.blocks[block_id];
        let mut next = Vec::with_capacity(beam.len() * params.beam.candidate_count.max(1));

        for (parent_index, state) in beam.iter().enumerate() {
            let Some(bounds) =
                entry_time_bounds_for_state(problem, precedence_context.as_ref(), state, block_id)
            else {
                continue;
            };
            let mut placements = Vec::with_capacity(params.beam.candidate_count);

            for template in templates {
                let Some(scheduled) =
                    materialize_template(problem, pre, state, block_id, template, bounds)
                else {
                    continue;
                };
                push_placement_candidate(
                    problem,
                    pre,
                    parent_index,
                    state,
                    original,
                    scheduled,
                    &mut placements,
                );
                if placements.len() >= params.beam.candidate_count {
                    break;
                }
            }
            if let Some(scheduled) =
                compatible_original_placement(problem, pre, state, original, bounds)
            {
                push_placement_candidate(
                    problem,
                    pre,
                    parent_index,
                    state,
                    original,
                    scheduled,
                    &mut placements,
                );
            }
            retain_parent_candidates(&mut placements, params.beam.candidate_count);

            next.extend(
                placements
                    .into_iter()
                    .filter(|placement| placement.score13 <= accept_threshold + 1e-9),
            );
        }

        let selected = select_beam(
            next,
            &beam,
            params.beam.width,
            params.beam.state_group_limit,
        );
        let mut next_beam = Vec::with_capacity(selected.len());
        for candidate in selected {
            let parent = &beam[candidate.parent_index];
            let mut added = parent.added.clone();
            added.push(candidate.scheduled);
            let mut projected_loads = parent.projected_loads.clone();
            let workload = block.workload as f64;
            projected_loads[original.bay_id] -= workload;
            projected_loads[candidate.scheduled.bay_id] += workload;
            next_beam.push(BeamState {
                added,
                projected_loads,
                score13: candidate.score13,
                exact_hash: candidate.exact_hash,
                placement_hash: candidate.placement_hash,
                changed: candidate.changed,
            });
        }
        beam = next_beam;
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

fn build_base_templates<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    base: &[ScheduledBlock],
    block_id: usize,
    original_orient_idx: usize,
    bounds: EntryTimeBounds,
    candidate_pool_count: usize,
    candidate_group_limit: usize,
    orientation_sample_count: usize,
    y_sample_count: usize,
    rng: &mut R,
) -> Vec<BaseInsertTemplate> {
    if candidate_pool_count == 0
        || candidate_group_limit == 0
        || orientation_sample_count == 0
        || y_sample_count == 0
    {
        return Vec::new();
    }

    let block = &problem.blocks[block_id];
    let mut orientation_indices: Vec<_> = (0..block.shape.len())
        .filter(|&orient_idx| orient_idx != original_orient_idx)
        .collect();
    rng.shuffle(&mut orientation_indices);
    orientation_indices.truncate(orientation_sample_count.saturating_sub(1));
    orientation_indices.push(original_orient_idx);
    let mut candidate_groups = Vec::new();

    for bay_id in 0..problem.bays.len() {
        let Some(mut scanner) =
            PlacementXScanner::new(problem, pre, base, block_id, bay_id, bounds.min, bounds.max)
        else {
            continue;
        };
        for &orient_idx in &orientation_indices {
            let Some(fit_range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            let mut ys: Vec<_> = (fit_range.min_y + 1..=fit_range.max_y).collect();
            rng.shuffle(&mut ys);
            ys.truncate(y_sample_count.saturating_sub(1));
            ys.push(fit_range.min_y);
            ys.sort_unstable();
            let per_y_limit = candidate_group_limit.div_ceil(ys.len());
            let mut y_groups = Vec::new();
            for y in ys {
                let mut y_group: Vec<BaseInsertTemplate> = Vec::new();
                scanner.scan_y_ranges(
                    orient_idx,
                    y,
                    fit_range.min_x,
                    fit_range.max_x,
                    |range, forbidden| {
                        let Some(sort_entry_time) =
                            forbidden.first_feasible_time(&[], bounds.min, bounds.max)
                        else {
                            return false;
                        };
                        let mut base_forbidden_intervals = Vec::new();
                        forbidden.collect_merged(&mut base_forbidden_intervals);
                        if let Some(template) = y_group.iter_mut().find(|template| {
                            template.base_forbidden_intervals == base_forbidden_intervals
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
                        y_group.push(BaseInsertTemplate {
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
                for template in &mut y_group {
                    merge_x_ranges(&mut template.x_ranges);
                }
                y_group.sort_by(base_template_cmp);
                y_group.truncate(per_y_limit);
                if !y_group.is_empty() {
                    y_groups.push(y_group);
                }
            }
            y_groups.sort_by(|a, b| base_template_cmp(&a[0], &b[0]));
            let mut y_iterators: Vec<_> = y_groups
                .into_iter()
                .map(|group| group.into_iter())
                .collect();
            let mut group = Vec::with_capacity(candidate_group_limit);
            while group.len() < candidate_group_limit {
                let mut progressed = false;
                for iter in &mut y_iterators {
                    if let Some(template) = iter.next() {
                        group.push(template);
                        progressed = true;
                        if group.len() >= candidate_group_limit {
                            break;
                        }
                    }
                }
                if !progressed {
                    break;
                }
            }
            if !group.is_empty() {
                candidate_groups.push(group);
            }
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
    let key = |template: &BaseInsertTemplate| {
        (
            template.sort_entry_time,
            template.bay_id,
            template.orient_idx,
            template.y,
            template.x_ranges[0].min_x,
        )
    };
    a.sort_score13
        .total_cmp(&b.sort_score13)
        .then_with(|| key(a).cmp(&key(b)))
}

fn compatible_original_placement(
    problem: &Problem,
    pre: &Precompute,
    state: &BeamState,
    original: ScheduledBlock,
    bounds: EntryTimeBounds,
) -> Option<ScheduledBlock> {
    if original.entry_time < bounds.min || original.entry_time > bounds.max {
        return None;
    }
    let range = XRange {
        min_x: original.x,
        max_x: original.x,
    };
    find_leftmost_fixed_time_x(
        problem,
        pre,
        original.block_id,
        original.bay_id,
        original.orient_idx,
        original.y,
        original.entry_time,
        range,
        &state.added,
    )
    .map(|_| original)
}

fn build_beam_precedence_context(
    problem: &Problem,
    constraints: &HeuristicPrecedence,
    base_by_id: &[Option<ScheduledBlock>],
    order: &[usize],
) -> Option<BeamPrecedenceContext> {
    let mut removed_position = vec![None; problem.blocks.len()];
    for (position, &block_id) in order.iter().enumerate() {
        removed_position[block_id] = Some(position);
    }

    let mut base_bounds: Vec<_> = problem
        .blocks
        .iter()
        .map(|block| EntryTimeBounds {
            min: block.release_time,
            max: i64::MAX,
        })
        .collect();
    let mut removed_predecessor_positions = vec![Vec::new(); problem.blocks.len()];
    for (position, &block_id) in order.iter().enumerate() {
        let mut min_entry_time = problem.blocks[block_id].release_time;
        for &before in &constraints.predecessors[block_id] {
            if let Some(before_position) = removed_position[before] {
                debug_assert!(before_position < position);
                removed_predecessor_positions[block_id].push(before_position);
            } else {
                min_entry_time = min_entry_time.max(base_by_id[before]?.exit_time);
            }
        }

        let mut max_entry_time = i64::MAX;
        let process_time = problem.blocks[block_id].processing_time;
        for &after in &constraints.successors[block_id] {
            if removed_position[after].is_none() {
                max_entry_time = max_entry_time.min(base_by_id[after]?.entry_time - process_time);
            }
        }
        if min_entry_time > max_entry_time {
            return None;
        }
        base_bounds[block_id] = EntryTimeBounds {
            min: min_entry_time,
            max: max_entry_time,
        };
    }

    Some(BeamPrecedenceContext {
        base_bounds,
        removed_predecessor_positions,
    })
}

fn entry_time_bounds_for_state(
    problem: &Problem,
    context: Option<&BeamPrecedenceContext>,
    state: &BeamState,
    block_id: usize,
) -> Option<EntryTimeBounds> {
    let Some(context) = context else {
        return Some(EntryTimeBounds {
            min: problem.blocks[block_id].release_time,
            max: i64::MAX,
        });
    };

    let mut bounds = context.base_bounds[block_id];
    for &position in &context.removed_predecessor_positions[block_id] {
        bounds.min = bounds.min.max(state.added.get(position)?.exit_time);
    }
    (bounds.min <= bounds.max).then_some(bounds)
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

fn push_placement_candidate(
    problem: &Problem,
    pre: &Precompute,
    parent_index: usize,
    state: &BeamState,
    original: ScheduledBlock,
    scheduled: ScheduledBlock,
    placements: &mut Vec<BeamCandidate>,
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
    placements.push(BeamCandidate {
        parent_index,
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

fn retain_parent_candidates(candidates: &mut Vec<BeamCandidate>, limit: usize) {
    candidates.sort_by(placement_candidate_cmp);
    candidates.dedup_by(|a, b| a.scheduled == b.scheduled);
    candidates.truncate(limit);
}

fn placement_candidate_cmp(a: &BeamCandidate, b: &BeamCandidate) -> std::cmp::Ordering {
    a.score
        .total_cmp(&b.score)
        .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
        .then(a.scheduled.bay_id.cmp(&b.scheduled.bay_id))
        .then(a.scheduled.orient_idx.cmp(&b.scheduled.orient_idx))
        .then(a.scheduled.y.cmp(&b.scheduled.y))
        .then(a.scheduled.x.cmp(&b.scheduled.x))
}

fn select_beam(
    mut candidates: Vec<BeamCandidate>,
    parents: &[BeamState],
    width: usize,
    group_limit: usize,
) -> Vec<BeamCandidate> {
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
        if contains_exact(&selected, &candidate, parents) {
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
        selected.push(candidate);
        if selected.len() >= width {
            break;
        }
    }
    if selected.len() < width {
        for candidate in deferred {
            if contains_exact(&selected, &candidate, parents) {
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

fn contains_exact(
    candidates: &[BeamCandidate],
    candidate: &BeamCandidate,
    parents: &[BeamState],
) -> bool {
    candidates.iter().any(|current| {
        current.exact_hash == candidate.exact_hash
            && current.scheduled == candidate.scheduled
            && parents[current.parent_index].added == parents[candidate.parent_index].added
    })
}

fn hash_block_placement(block: ScheduledBlock) -> u64 {
    let mut hash = HASH_OFFSET;
    hash = mix_hash(hash, block.block_id as u64);
    hash = mix_hash(hash, block.bay_id as u64);
    hash = mix_hash(hash, block.orient_idx as u64);
    mix_hash(hash, block.y as u64)
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
