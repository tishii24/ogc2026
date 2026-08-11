use crate::{
    INF, Problem, ScheduledBlock,
    params::{InsertAnchor, InsertParams, RotateNeighborParams, ShiftNeighborParams},
    utils::random::Random,
};

use super::{
    insert::insert_greedy,
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

fn horizontal_anchor(anchor: InsertAnchor) -> HorizontalAnchor {
    match anchor {
        InsertAnchor::BottomLeft | InsertAnchor::TopLeft => HorizontalAnchor::Left,
        InsertAnchor::BottomRight | InsertAnchor::TopRight => HorizontalAnchor::Right,
    }
}

fn anchor_cmp(
    pre: &Precompute,
    a: ScheduledBlock,
    b: ScheduledBlock,
    anchor: InsertAnchor,
) -> std::cmp::Ordering {
    let a_bounds = pre.orientation_bbox_bounds[a.block_id][a.orient_idx];
    let b_bounds = pre.orientation_bbox_bounds[b.block_id][b.orient_idx];
    match anchor {
        InsertAnchor::BottomLeft => (a.x as f64 + a_bounds.max_x)
            .total_cmp(&(b.x as f64 + b_bounds.max_x))
            .then((a.y as f64 + a_bounds.max_y).total_cmp(&(b.y as f64 + b_bounds.max_y))),
        InsertAnchor::BottomRight => (b.x as f64 + b_bounds.min_x)
            .total_cmp(&(a.x as f64 + a_bounds.min_x))
            .then((a.y as f64 + a_bounds.max_y).total_cmp(&(b.y as f64 + b_bounds.max_y))),
        InsertAnchor::TopLeft => (a.x as f64 + a_bounds.max_x)
            .total_cmp(&(b.x as f64 + b_bounds.max_x))
            .then((b.y as f64 + b_bounds.min_y).total_cmp(&(a.y as f64 + a_bounds.min_y))),
        InsertAnchor::TopRight => (b.x as f64 + b_bounds.min_x)
            .total_cmp(&(a.x as f64 + a_bounds.min_x))
            .then((b.y as f64 + b_bounds.min_y).total_cmp(&(a.y as f64 + a_bounds.min_y))),
    }
}

fn update_best(
    problem: &Problem,
    pre: &Precompute,
    anchor: InsertAnchor,
    best: &mut Option<(i64, i64, ScheduledBlock)>,
    candidate: ScheduledBlock,
) {
    let block = &problem.blocks[candidate.block_id];
    let tardiness = (candidate.exit_time - block.due_date).max(0);
    if best
        .as_ref()
        .is_none_or(|&(best_tardiness, best_entry_time, best_scheduled)| {
            (tardiness, candidate.entry_time)
                .cmp(&(best_tardiness, best_entry_time))
                .then_with(|| anchor_cmp(pre, candidate, best_scheduled, anchor))
                .is_lt()
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
    anchor: InsertAnchor,
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
    let dy_range = match anchor {
        InsertAnchor::BottomLeft | InsertAnchor::BottomRight => params.dy_range,
        InsertAnchor::TopLeft | InsertAnchor::TopRight => (-params.dy_range.1, -params.dy_range.0),
    };
    let horizontal_anchor = horizontal_anchor(anchor);
    for dy in dy_range.0..=dy_range.1 {
        let y = old.y + dy;
        scanner.scan_y(old.orient_idx, y, horizontal_anchor, |moved| {
            if moved != old {
                update_best(problem, pre, anchor, &mut best, moved);
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
    anchor: InsertAnchor,
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
    let horizontal_anchor = horizontal_anchor(anchor);
    for neighbor in &pre.orientation_neighbors[old.block_id][old.orient_idx] {
        for ddy in params.dy_range.0..=params.dy_range.1 {
            let y = old.y + neighbor.dy + ddy;
            scanner.scan_y(neighbor.orient_idx, y, horizontal_anchor, |rotated| {
                update_best(problem, pre, anchor, &mut best, rotated);
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
    anchor: InsertAnchor,
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
