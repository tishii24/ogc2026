use crate::{
    Problem, ScheduledBlock,
    params::{
        InsertParams, MoveNeighborParams, RotateNeighborParams, ShiftNeighborParams,
        SwapNeighborParams,
    },
    utils::random::Random,
};

use super::{
    insert::insert_greedy,
    objective::{ScoreWeights, score13_block_weighted},
    placement_scan::PlacementXScanner,
    precompute::Precompute,
    reconstruct::{
        EntryTimeBounds, HeuristicPrecedence, precedence_entry_time_bounds, scheduled_by_id,
    },
};

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(super) enum NeighborKind {
    LargeReconstruct,
    Shift,
    Move,
    Rotate,
    Swap,
}

impl NeighborKind {
    pub(super) const ALL: [Self; 5] = [
        Self::LargeReconstruct,
        Self::Shift,
        Self::Move,
        Self::Rotate,
        Self::Swap,
    ];
    pub(super) const NAMES: &'static [&'static str] = &["Large", "Shift", "Move", "Rotate", "Swap"];

    pub(super) fn index(self) -> usize {
        self as usize
    }
}

pub(super) fn sample_neighbor<R: Random>(rng: &mut R, weights: &[f64; 5]) -> NeighborKind {
    let total = weights.iter().sum::<f64>();
    debug_assert!(total > 0.0);

    let mut value = rng.next_f64() * total;
    for (index, &weight) in weights.iter().enumerate() {
        if value < weight {
            return NeighborKind::ALL[index];
        }
        value -= weight;
    }
    NeighborKind::ALL[weights.iter().rposition(|&weight| weight > 0.0).unwrap()]
}

fn update_best(
    problem: &Problem,
    best: &mut Option<(i64, i64, ScheduledBlock)>,
    candidate: ScheduledBlock,
) {
    let block = &problem.blocks[candidate.block_id];
    let tardiness = (candidate.exit_time - block.due_date).max(0);
    if best
        .as_ref()
        .is_none_or(|&(best_tardiness, best_entry_time, _)| {
            (tardiness, candidate.entry_time) < (best_tardiness, best_entry_time)
        })
    {
        *best = Some((tardiness, candidate.entry_time, candidate));
    }
}

pub(super) fn try_shift_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    constraints: Option<&HeuristicPrecedence>,
    params: &ShiftNeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];

    let mut base = Vec::with_capacity(schedule.len());
    for (i, &s) in schedule.iter().enumerate() {
        if i != idx {
            base.push(s);
        }
    }

    let EntryTimeBounds {
        min: min_entry_time,
        max: max_entry_time,
    } = if let Some(constraints) = constraints {
        let by_id = scheduled_by_id(problem, &base);
        precedence_entry_time_bounds(problem, constraints, &by_id, old.block_id)?
    } else {
        EntryTimeBounds {
            min: i64::MIN,
            max: i64::MAX,
        }
    };
    let mut scanner = PlacementXScanner::new(
        problem,
        pre,
        &base,
        old.block_id,
        old.bay_id,
        min_entry_time,
        max_entry_time,
    )?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;
    for dy in params.dy_range.0..=params.dy_range.1 {
        let y = old.y + dy;
        scanner.scan_y(old.orient_idx, y, |moved| {
            if moved != old {
                update_best(problem, &mut best, moved);
            }
            false
        });
    }

    let moved = best?.2;
    base.push(moved);
    Some(base)
}

pub(super) fn try_rotate_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    constraints: Option<&HeuristicPrecedence>,
    params: &RotateNeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_index(schedule.len());
    let old = schedule[idx];
    let block = &problem.blocks[old.block_id];
    if block.shape.len() <= 1 {
        return None;
    }

    let mut base = Vec::with_capacity(schedule.len());
    for (i, &s) in schedule.iter().enumerate() {
        if i != idx {
            base.push(s);
        }
    }

    let EntryTimeBounds {
        min: min_entry_time,
        max: max_entry_time,
    } = if let Some(constraints) = constraints {
        let by_id = scheduled_by_id(problem, &base);
        precedence_entry_time_bounds(problem, constraints, &by_id, old.block_id)?
    } else {
        EntryTimeBounds {
            min: i64::MIN,
            max: i64::MAX,
        }
    };
    let mut scanner = PlacementXScanner::new(
        problem,
        pre,
        &base,
        old.block_id,
        old.bay_id,
        min_entry_time,
        max_entry_time,
    )?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;
    for neighbor in &pre.orientation_neighbors[old.block_id][old.orient_idx] {
        for ddy in params.dy_range.0..=params.dy_range.1 {
            let y = old.y + neighbor.dy + ddy;
            scanner.scan_y(neighbor.orient_idx, y, |rotated| {
                update_best(problem, &mut best, rotated);
                false
            });
        }
    }

    let rotated = best?.2;
    base.push(rotated);
    Some(base)
}

fn try_swap_place(
    problem: &Problem,
    pre: &Precompute,
    old: ScheduledBlock,
    schedule: &[ScheduledBlock],
    bay_id: usize,
    orient_idx: usize,
    base_y: i64,
    min_entry_time: i64,
    max_entry_time: i64,
    params: &SwapNeighborParams,
) -> Option<ScheduledBlock> {
    let mut scanner = PlacementXScanner::new(
        problem,
        pre,
        schedule,
        old.block_id,
        bay_id,
        min_entry_time,
        max_entry_time,
    )?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;
    for dy in params.dy_range.0..=params.dy_range.1 {
        scanner.scan_y(orient_idx, base_y + dy, |scheduled| {
            update_best(problem, &mut best, scheduled);
            false
        });
    }

    Some(best?.2)
}

pub(super) fn try_swap_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    constraints: Option<&HeuristicPrecedence>,
    params: &SwapNeighborParams,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.len() < 2 {
        return None;
    }

    let a_idx = rng.gen_index(schedule.len());
    let a_old = schedule[a_idx];
    let candidates: Vec<_> = pre.swap_neighbors[a_old.block_id][a_old.orient_idx]
        .iter()
        .filter_map(|&candidate| {
            let b_idx = schedule
                .iter()
                .position(|s| s.block_id == candidate.block_id)?;
            (constraints.is_none() || schedule[b_idx].bay_id == a_old.bay_id)
                .then_some((candidate, b_idx))
        })
        .take(params.neighbor_top_k)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let (candidate, b_idx) = candidates[rng.gen_index(candidates.len())];
    let b_old = schedule[b_idx];

    let mut cur = Vec::with_capacity(schedule.len());
    for (idx, &s) in schedule.iter().enumerate() {
        if idx != a_idx && idx != b_idx {
            cur.push(s);
        }
    }

    let mut targets = [
        (
            a_old,
            b_old.bay_id,
            a_old.orient_idx,
            b_old.y - candidate.dy,
        ),
        (
            b_old,
            a_old.bay_id,
            candidate.orient_idx,
            a_old.y + candidate.dy,
        ),
    ];
    if constraints.is_some_and(|constraints| {
        constraints.predecessors[a_old.block_id].contains(&b_old.block_id)
    }) {
        targets.swap(0, 1);
    }

    for (old, bay_id, orient_idx, base_y) in targets {
        let EntryTimeBounds {
            min: min_entry_time,
            max: max_entry_time,
        } = if let Some(constraints) = constraints {
            let by_id = scheduled_by_id(problem, &cur);
            precedence_entry_time_bounds(problem, constraints, &by_id, old.block_id)?
        } else {
            EntryTimeBounds {
                min: i64::MIN,
                max: i64::MAX,
            }
        };
        let new = try_swap_place(
            problem,
            pre,
            old,
            &cur,
            bay_id,
            orient_idx,
            base_y,
            min_entry_time,
            max_entry_time,
            params,
        )?;
        cur.push(new);
    }

    Some(cur)
}

pub(super) fn try_move_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    constraints: Option<&HeuristicPrecedence>,
    params: &MoveNeighborParams,
    insert_params: &InsertParams,
    score_weights: ScoreWeights,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let sample_count = params.sample_blocks.min(schedule.len());
    let mut indices: Vec<usize> = (0..schedule.len()).collect();
    rng.shuffle(&mut indices);
    indices.truncate(sample_count);
    indices.sort_by(|&a, &b| {
        pre.max_footprint_area[schedule[a].block_id]
            .total_cmp(&pre.max_footprint_area[schedule[b].block_id])
            .then(schedule[a].block_id.cmp(&schedule[b].block_id))
    });
    indices.truncate(params.small_pool_size.min(indices.len()));

    let idx = indices.into_iter().max_by(|&a, &b| {
        let sa = score13_block_weighted(problem, pre, schedule[a], score_weights);
        let sb = score13_block_weighted(problem, pre, schedule[b], score_weights);
        sa.total_cmp(&sb).then(
            pre.max_footprint_area[schedule[b].block_id]
                .total_cmp(&pre.max_footprint_area[schedule[a].block_id]),
        )
    })?;
    let old = schedule[idx];

    let mut base = Vec::with_capacity(schedule.len());
    let mut loads = vec![0.0; problem.bays.len()];
    for (i, &s) in schedule.iter().enumerate() {
        if i == idx {
            continue;
        }
        loads[s.bay_id] += problem.blocks[s.block_id].workload as f64;
        base.push(s);
    }

    let EntryTimeBounds {
        min: min_entry_time,
        max: max_entry_time,
    } = if let Some(constraints) = constraints {
        let by_id = scheduled_by_id(problem, &base);
        precedence_entry_time_bounds(problem, constraints, &by_id, old.block_id)?
    } else {
        EntryTimeBounds {
            min: i64::MIN,
            max: i64::MAX,
        }
    };
    let scheduled = insert_greedy(
        problem,
        pre,
        old,
        min_entry_time,
        max_entry_time,
        &base,
        &loads,
        insert_params,
        score_weights,
        &pre.bay_order_by_pref[old.block_id],
        1,
        1.0,
        rng,
    )?;
    if scheduled == old {
        return None;
    }

    base.push(scheduled);
    Some(base)
}
