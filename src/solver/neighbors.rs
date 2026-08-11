use crate::{
    INF, Problem, ScheduledBlock,
    params::{InsertParams, RotateNeighborParams, ShiftNeighborParams},
    utils::random::Random,
};

use super::{
    insert::{insert_greedy, sample_insert_anchor},
    placement_scan::{HorizontalAnchor, PlacementXScanner},
    precompute::Precompute,
};

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
pub(super) enum NeighborKind {
    LargeReconstruct,
    Shift,
    Move,
    Rotate,
}

impl NeighborKind {
    pub(super) const ALL: [Self; 4] = [
        Self::LargeReconstruct,
        Self::Shift,
        Self::Move,
        Self::Rotate,
    ];
    pub(super) const NAMES: &'static [&'static str] = &["Large", "Shift", "Move", "Rotate"];

    pub(super) fn index(self) -> usize {
        self as usize
    }
}

pub(super) fn sample_neighbor<R: Random>(rng: &mut R, weights: &[f64; 4]) -> NeighborKind {
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

    let mut scanner =
        PlacementXScanner::new(problem, pre, &base, old.block_id, old.bay_id, -INF, INF)?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;
    for dy in params.dy_range.0..=params.dy_range.1 {
        let y = old.y + dy;
        scanner.scan_y(old.orient_idx, y, HorizontalAnchor::Left, |moved| {
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

    let mut scanner =
        PlacementXScanner::new(problem, pre, &base, old.block_id, old.bay_id, -INF, INF)?;
    let mut best: Option<(i64, i64, ScheduledBlock)> = None;
    for neighbor in &pre.orientation_neighbors[old.block_id][old.orient_idx] {
        for ddy in params.dy_range.0..=params.dy_range.1 {
            let y = old.y + neighbor.dy + ddy;
            scanner.scan_y(neighbor.orient_idx, y, HorizontalAnchor::Left, |rotated| {
                update_best(problem, &mut best, rotated);
                false
            });
        }
    }

    let rotated = best?.2;
    base.push(rotated);
    Some(base)
}

pub(super) fn try_move_neighbor<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    schedule: &[ScheduledBlock],
    rng: &mut R,
    insert_params: &InsertParams,
    w2: f64,
) -> Option<Vec<ScheduledBlock>> {
    if schedule.is_empty() {
        return None;
    }

    let idx = rng.gen_range(0, schedule.len());
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

    let anchor = sample_insert_anchor(rng, insert_params);
    let scheduled = insert_greedy(
        problem,
        pre,
        old,
        -INF,
        old.entry_time,
        &base,
        &loads,
        insert_params,
        &pre.bay_order_by_pref[old.block_id],
        1,
        1.0,
        w2,
        anchor,
        rng,
    )?;
    if scheduled == old {
        return None;
    }

    base.push(scheduled);
    Some(base)
}
