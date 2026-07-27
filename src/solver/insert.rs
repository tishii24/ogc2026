use crate::{Problem, ScheduledBlock, params::InsertParams, utils::random::Random};

use super::{
    objective::normalized_imbalance, placement_scan::PlacementXScanner, precompute::Precompute,
};

struct InsertCandidate {
    scheduled: ScheduledBlock,
    score_delta: f64,
    bbox_left: f64,
    bbox_right: f64,
    bbox_bottom: f64,
    bbox_top: f64,
}

#[derive(Clone, Copy)]
enum BBoxAnchor {
    RightTop,
    RightBottom,
    LeftTop,
    LeftBottom,
}

pub(crate) fn insert_greedy<R: Random>(
    problem: &Problem,
    pre: &Precompute,
    original: ScheduledBlock,
    min_entry_time: i64,
    max_entry_time: i64,
    schedule: &[ScheduledBlock],
    loads: &[f64],
    params: &InsertParams,
    reconstruct_random_strength: Option<f64>,
    bay_order: &[usize],
    rng: &mut R,
) -> Option<ScheduledBlock> {
    fn insert_candidate_better(
        a: &InsertCandidate,
        b: &InsertCandidate,
        anchor: BBoxAnchor,
    ) -> bool {
        let bbox_order = match anchor {
            BBoxAnchor::RightTop => a
                .bbox_right
                .total_cmp(&b.bbox_right)
                .then(a.bbox_top.total_cmp(&b.bbox_top)),
            BBoxAnchor::RightBottom => a
                .bbox_right
                .total_cmp(&b.bbox_right)
                .then(b.bbox_bottom.total_cmp(&a.bbox_bottom)),
            BBoxAnchor::LeftTop => b
                .bbox_left
                .total_cmp(&a.bbox_left)
                .then(a.bbox_top.total_cmp(&b.bbox_top)),
            BBoxAnchor::LeftBottom => b
                .bbox_left
                .total_cmp(&a.bbox_left)
                .then(b.bbox_bottom.total_cmp(&a.bbox_bottom)),
        };
        a.score_delta
            .total_cmp(&b.score_delta)
            .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
            .then(bbox_order)
            .then(a.scheduled.block_id.cmp(&b.scheduled.block_id))
            .is_lt()
    }

    let block_id = original.block_id;
    let block = &problem.blocks[block_id];
    let process_t = block.processing_time;
    let min_t = block.release_time.max(min_entry_time);
    let max_t = max_entry_time;
    if min_t > max_t {
        return None;
    }

    let current_obj2 = normalized_imbalance(loads, &pre.bay_load_scale);
    let original_tardiness = (original.exit_time - block.due_date).max(0);
    let anchor = match reconstruct_random_strength {
        Some(strength)
            if strength > 0.0
                && rng.next_f64() < params.reconstruct_bbox_anchor_probability * strength =>
        {
            match rng.gen_index(4) {
                0 => BBoxAnchor::RightTop,
                1 => BBoxAnchor::RightBottom,
                2 => BBoxAnchor::LeftTop,
                _ => BBoxAnchor::LeftBottom,
            }
        }
        _ => BBoxAnchor::RightTop,
    };
    let mut best: Option<InsertCandidate> = None;

    for &bay_id in bay_order {
        if reconstruct_random_strength.is_some_and(|strength| {
            strength > 0.0
                && bay_id != original.bay_id
                && rng.next_f64() < params.reconstruct_bay_skip_probability * strength
        }) {
            continue;
        }

        let mut next_loads = loads.to_vec();
        next_loads[bay_id] += block.workload as f64;

        let delta_obj23 = problem.weights.w2
            * (normalized_imbalance(&next_loads, &pre.bay_load_scale) - current_obj2)
            + problem.weights.w3 * pre.pref_penalty[block_id][bay_id] as f64;
        let min_tardiness = min_t
            .saturating_add(process_t)
            .saturating_sub(block.due_date)
            .max(0);
        let lower_score_delta = problem.weights.w1 * min_tardiness as f64 + delta_obj23;
        if best
            .as_ref()
            .is_some_and(|best| lower_score_delta > best.score_delta)
        {
            continue;
        }

        let mut scanner =
            PlacementXScanner::new(problem, pre, schedule, block_id, bay_id, min_t, max_t)?;

        for &orient_idx in &pre.orientation_order_by_bbox[block_id] {
            if reconstruct_random_strength.is_some_and(|strength| {
                strength > 0.0
                    && (bay_id != original.bay_id || orient_idx != original.orient_idx)
                    && rng.next_f64() < params.reconstruct_orientation_skip_probability * strength
            }) {
                continue;
            }
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            let bounds = pre.orientation_bbox_bounds[block_id][orient_idx];

            let mut ys: Vec<i64> = (range.min_y..=range.max_y).collect();
            rng.shuffle(&mut ys);
            if let Some(strength) = reconstruct_random_strength {
                let original_y = (bay_id == original.bay_id
                    && orient_idx == original.orient_idx
                    && range.min_y <= original.y
                    && original.y <= range.max_y)
                    .then_some(original.y);
                let skip_probability = params.reconstruct_y_skip_probability * strength;
                if skip_probability > 0.0 {
                    ys.retain(|&y| Some(y) == original_y || rng.next_f64() >= skip_probability);
                }
            }
            let mut remaining_y_buffer = None;

            for y in ys {
                if remaining_y_buffer == Some(0) {
                    break;
                }

                let valid_y = scanner.scan_y(orient_idx, y, |scheduled| {
                    let tardiness = (scheduled.exit_time - block.due_date).max(0);
                    let score_delta = problem.weights.w1 * tardiness as f64 + delta_obj23;
                    let candidate = InsertCandidate {
                        scheduled,
                        score_delta,
                        bbox_left: scheduled.x as f64 + bounds.min_x,
                        bbox_right: scheduled.x as f64 + bounds.max_x,
                        bbox_bottom: scheduled.y as f64 + bounds.min_y,
                        bbox_top: scheduled.y as f64 + bounds.max_y,
                    };
                    if best.as_ref().map_or(true, |best| {
                        insert_candidate_better(&candidate, best, anchor)
                    }) {
                        best = Some(candidate);
                    }
                    tardiness <= original_tardiness
                });

                match &mut remaining_y_buffer {
                    None if valid_y => remaining_y_buffer = Some(params.y_buffer),
                    Some(remaining) => *remaining -= 1,
                    None => {}
                }
            }
        }
    }

    let result = best.map(|candidate| candidate.scheduled);
    result
}
