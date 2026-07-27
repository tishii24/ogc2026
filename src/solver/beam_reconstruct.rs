use super::{
    PrecedenceConstraints, build_topological_order, choose_removed_blocks,
    precedence_entry_time_range, sample_reconstruct_order_weights, sample_removed_count,
    scheduled_by_id, score13_block, sort_block_order,
};
use crate::{
    Problem, ScheduledBlock,
    solver::{PlacementXScanner, insert::Interval},
    utils::{base::rand::Random, params::NeighborParams, precompute::Precompute},
};

const HASH_OFFSET: u64 = 1469598103934665603;

struct BaseInsertCandidate {
    bay_id: usize,
    orient_idx: usize,
    y: i64,
    min_x: i64,
    max_x: i64,
    entry_time: i64,
    score13: f64,
    forbidden_intervals: Vec<Interval>,
}

#[derive(Clone)]
struct BeamState {
    added: Vec<ScheduledBlock>,
    loads: Vec<f64>,
    score13: f64,
    score: f64,
    exact_hash: u64,
    placement_hash: u64,
    changed: bool,
    original_prefix: bool,
}

#[derive(Clone, Copy)]
struct PlacementCandidate {
    scheduled: ScheduledBlock,
    score13: f64,
    score: f64,
    exact_hash: u64,
    placement_hash: u64,
    changed: bool,
    original_prefix: bool,
}

pub(super) fn try_beam_large_reconstruct<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    constraints: Option<&PrecedenceConstraints>,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    accept_threshold: f64,
    params: &NeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    let k = sample_removed_count(rng, params).min(problem.blocks.len());
    let mut removed_ids = choose_removed_blocks(problem, pre, schedule, k, rng, params)?;
    if removed_ids.is_empty() {
        return None;
    }

    let weights = sample_reconstruct_order_weights(rng, params);
    let order = if let Some(constraints) = constraints {
        build_topological_order(problem, pre, &removed_ids, constraints, weights, rng)
    } else {
        sort_block_order(
            problem,
            &pre.block_area,
            &pre.pref_spread,
            &mut removed_ids,
            weights,
            rng,
        );
        removed_ids.clone()
    };

    let original_by_id = scheduled_by_id(problem, schedule);
    let mut removed = vec![false; problem.blocks.len()];
    for &block_id in &removed_ids {
        removed[block_id] = true;
    }

    let mut base = Vec::with_capacity(schedule.len() - removed_ids.len());
    let mut base_loads = vec![0.0; problem.bays.len()];
    let mut base_score13 = 0.0;
    for &scheduled in schedule {
        if removed[scheduled.block_id] {
            continue;
        }
        base_loads[scheduled.bay_id] += problem.blocks[scheduled.block_id].workload as f64;
        base_score13 += score13_block(problem, pre, scheduled);
        base.push(scheduled);
    }
    if base_score13 > accept_threshold + 1e-9 {
        return None;
    }

    let mut candidates_by_id: Vec<Vec<BaseInsertCandidate>> =
        (0..problem.blocks.len()).map(|_| Vec::new()).collect();
    for &block_id in &order {
        candidates_by_id[block_id] = build_base_candidates(
            problem,
            pre,
            &base,
            block_id,
            params.beam_large_reconstruct.candidate_count,
            params.beam_large_reconstruct.placement_group_limit,
        );
    }

    let base_by_id = scheduled_by_id(problem, &base);
    let initial_score = base_score13 + problem.weights.w2 * normalized_imbalance(pre, &base_loads);
    let mut beam = vec![BeamState {
        added: Vec::with_capacity(order.len()),
        loads: base_loads,
        score13: base_score13,
        score: initial_score,
        exact_hash: 0,
        placement_hash: 0,
        changed: false,
        original_prefix: true,
    }];

    for &block_id in &order {
        let original = original_by_id[block_id]?;
        let block = &problem.blocks[block_id];
        let mut next =
            Vec::with_capacity(beam.len() * params.beam_large_reconstruct.candidate_count.max(1));

        for state in &beam {
            let (min_entry_time, max_entry_time) = if let Some(constraints) = constraints {
                let mut current_by_id = base_by_id.clone();
                for &scheduled in &state.added {
                    current_by_id[scheduled.block_id] = Some(scheduled);
                }
                let Some(range) =
                    precedence_entry_time_range(problem, constraints, &current_by_id, block_id)
                else {
                    continue;
                };
                range
            } else {
                (i64::MIN, i64::MAX)
            };

            let mut scanners: Vec<_> = (0..problem.bays.len())
                .map(|bay_id| {
                    PlacementXScanner::new(
                        problem,
                        pre,
                        &state.added,
                        block_id,
                        bay_id,
                        min_entry_time,
                        max_entry_time,
                    )
                })
                .collect();
            let mut placements = Vec::new();

            for candidate in &candidates_by_id[block_id] {
                let Some(scanner) = scanners[candidate.bay_id].as_mut() else {
                    continue;
                };
                scanner.scan_y_ranges(
                    candidate.orient_idx,
                    candidate.y,
                    candidate.min_x,
                    candidate.max_x,
                    |range, forbidden| {
                        let Some(entry_time) = forbidden.first_feasible_time(
                            &candidate.forbidden_intervals,
                            min_entry_time.max(block.release_time),
                            max_entry_time,
                        ) else {
                            return false;
                        };
                        let xs = [range.min_x, range.max_x];
                        for (index, x) in xs.into_iter().enumerate() {
                            if index == 1 && x == range.min_x {
                                continue;
                            }
                            push_placement_candidate(
                                problem,
                                pre,
                                state,
                                original,
                                ScheduledBlock {
                                    block_id,
                                    bay_id: candidate.bay_id,
                                    orient_idx: candidate.orient_idx,
                                    x,
                                    y: candidate.y,
                                    entry_time,
                                    exit_time: entry_time + block.processing_time,
                                },
                                &mut placements,
                            );
                        }
                        false
                    },
                );
            }

            if state.original_prefix {
                push_placement_candidate(problem, pre, state, original, original, &mut placements);
            }
            retain_parent_candidates(
                &mut placements,
                params.beam_large_reconstruct.candidate_count,
            );

            for placement in placements {
                if placement.score13 > accept_threshold + 1e-9 {
                    continue;
                }
                let mut added = state.added.clone();
                added.push(placement.scheduled);
                let mut loads = state.loads.clone();
                loads[placement.scheduled.bay_id] += block.workload as f64;
                next.push(BeamState {
                    added,
                    loads,
                    score13: placement.score13,
                    score: placement.score,
                    exact_hash: placement.exact_hash,
                    placement_hash: placement.placement_hash,
                    changed: placement.changed,
                    original_prefix: placement.original_prefix,
                });
            }
        }

        beam = select_beam(
            next,
            params.beam_large_reconstruct.beam_width,
            params.beam_large_reconstruct.placement_group_limit,
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
        let (score, _) = super::score_schedule(problem, pre, &blocks);
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

fn build_base_candidates(
    problem: &Problem,
    pre: &Precompute,
    base: &[ScheduledBlock],
    block_id: usize,
    candidate_count: usize,
    group_limit: usize,
) -> Vec<BaseInsertCandidate> {
    if candidate_count == 0 || group_limit == 0 {
        return Vec::new();
    }

    let block = &problem.blocks[block_id];
    let mut groups: Vec<Vec<Vec<BaseInsertCandidate>>> = (0..problem.bays.len())
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
            for y in fit_range.min_y..=fit_range.max_y {
                scanner.scan_y_ranges(
                    orient_idx,
                    y,
                    fit_range.min_x,
                    fit_range.max_x,
                    |range, forbidden| {
                        let Some(entry_time) =
                            forbidden.first_feasible_time(&[], block.release_time, i64::MAX)
                        else {
                            return false;
                        };
                        let scheduled = ScheduledBlock {
                            block_id,
                            bay_id,
                            orient_idx,
                            x: range.min_x,
                            y,
                            entry_time,
                            exit_time: entry_time + block.processing_time,
                        };
                        let score13 = score13_block(problem, pre, scheduled);
                        let group = &mut groups[bay_id][orient_idx];
                        if group.len() >= group_limit
                            && group.iter().all(|current| {
                                !base_candidate_better_values(
                                    score13,
                                    entry_time,
                                    y,
                                    range.min_x,
                                    current,
                                )
                            })
                        {
                            return false;
                        }

                        let mut forbidden_intervals = Vec::new();
                        forbidden.collect_merged(&mut forbidden_intervals);
                        group.push(BaseInsertCandidate {
                            bay_id,
                            orient_idx,
                            y,
                            min_x: range.min_x,
                            max_x: range.max_x,
                            entry_time,
                            score13,
                            forbidden_intervals,
                        });
                        group.sort_by(base_candidate_cmp);
                        group.truncate(group_limit);
                        false
                    },
                );
            }
        }
    }

    let mut result: Vec<_> = groups.into_iter().flatten().flatten().collect();
    result.sort_by(base_candidate_cmp);
    result.truncate(candidate_count);
    result
}

fn base_candidate_better_values(
    score13: f64,
    entry_time: i64,
    y: i64,
    x: i64,
    other: &BaseInsertCandidate,
) -> bool {
    score13
        .total_cmp(&other.score13)
        .then(entry_time.cmp(&other.entry_time))
        .then(y.cmp(&other.y))
        .then(x.cmp(&other.min_x))
        .is_lt()
}

fn base_candidate_cmp(a: &BaseInsertCandidate, b: &BaseInsertCandidate) -> std::cmp::Ordering {
    a.score13
        .total_cmp(&b.score13)
        .then(a.entry_time.cmp(&b.entry_time))
        .then(a.bay_id.cmp(&b.bay_id))
        .then(a.orient_idx.cmp(&b.orient_idx))
        .then(a.y.cmp(&b.y))
        .then(a.min_x.cmp(&b.min_x))
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
    let imbalance =
        normalized_imbalance_with_add(pre, &state.loads, scheduled.bay_id, block.workload as f64);
    placements.push(PlacementCandidate {
        scheduled,
        score13,
        score: score13 + problem.weights.w2 * imbalance,
        exact_hash: state.exact_hash ^ hash_scheduled_block(scheduled),
        placement_hash: state.placement_hash ^ hash_block_bay(scheduled),
        changed: state.changed || scheduled != original,
        original_prefix: state.original_prefix && scheduled == original,
    });
}

fn normalized_imbalance(pre: &Precompute, loads: &[f64]) -> f64 {
    normalized_imbalance_with_add(pre, loads, 0, 0.0)
}

fn normalized_imbalance_with_add(
    pre: &Precompute,
    loads: &[f64],
    added_bay: usize,
    added_load: f64,
) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let load = load + if bay_id == added_bay { added_load } else { 0.0 };
        let normalized = load * pre.bay_load_scale[bay_id];
        min_value = min_value.min(normalized);
        max_value = max_value.max(normalized);
    }
    (max_value - min_value).floor()
}

fn retain_parent_candidates(candidates: &mut Vec<PlacementCandidate>, limit: usize) {
    candidates.sort_by(placement_candidate_cmp);
    candidates.dedup_by(|a, b| a.scheduled == b.scheduled);
    if candidates.len() <= limit {
        return;
    }
    if let Some(original) = candidates
        .iter()
        .position(|candidate| candidate.original_prefix)
    {
        let original = candidates.remove(original);
        candidates.truncate(limit.saturating_sub(1));
        candidates.push(original);
    } else {
        candidates.truncate(limit);
    }
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

fn select_beam(mut candidates: Vec<BeamState>, width: usize, group_limit: usize) -> Vec<BeamState> {
    if width == 0 {
        return Vec::new();
    }
    candidates.sort_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then(a.exact_hash.cmp(&b.exact_hash))
    });

    let original = candidates
        .iter()
        .position(|state| state.original_prefix)
        .map(|index| candidates.remove(index));
    let normal_limit = width.saturating_sub(usize::from(original.is_some()));
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
        if group_count < group_limit && selected.len() < normal_limit {
            selected.push(candidate);
        } else {
            deferred.push(candidate);
        }
    }
    for candidate in deferred {
        if selected.len() >= normal_limit {
            break;
        }
        if !contains_exact(&selected, &candidate) {
            selected.push(candidate);
        }
    }
    if let Some(original) = original {
        if !contains_exact(&selected, &original) {
            selected.push(original);
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

fn hash_block_bay(block: ScheduledBlock) -> u64 {
    let mut hash = HASH_OFFSET;
    hash = mix_hash(hash, block.block_id as u64);
    hash = mix_hash(hash, block.bay_id as u64);
    // hash = mix_hash(hash, block.orient_idx as u64);
    hash
}

fn hash_scheduled_block(block: ScheduledBlock) -> u64 {
    let mut hash = hash_block_bay(block);
    hash = mix_hash(hash, block.x as u64);
    hash = mix_hash(hash, block.y as u64);
    mix_hash(hash, block.entry_time as u64)
}

fn mix_hash(mut hash: u64, value: u64) -> u64 {
    hash ^= value;
    hash.wrapping_mul(1099511628211)
}
