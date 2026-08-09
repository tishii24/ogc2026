#[cfg(feature = "profile-reconstruct")]
use std::time::{Duration, Instant};

use crate::{Boundsf, Problem, ScheduledBlock, params::InsertParams, utils::random::Random};

#[cfg(feature = "profile-reconstruct")]
use super::placement_scan::PlacementScanProfile;
use super::{objective::score_z2, placement_scan::PlacementXScanner, precompute::Precompute};

#[cfg(feature = "profile-reconstruct")]
#[derive(Default)]
pub(super) struct InsertGreedyProfile {
    pub(super) setup: Duration,
    pub(super) bay_scoring: Duration,
    pub(super) scanner_new: Duration,
    pub(super) orientation_prepare: Duration,
    pub(super) scan_y: Duration,
    pub(super) finalize: Duration,
    pub(super) calls: u64,
    pub(super) successes: u64,
    pub(super) bays: u64,
    pub(super) scanners: u64,
    pub(super) orientations: u64,
    pub(super) y_values: u64,
    pub(super) scan_y_calls: u64,
    pub(super) candidates: u64,
    pub(super) placement_scan: PlacementScanProfile,
}

struct InsertCandidate {
    scheduled: ScheduledBlock,
    score_delta: f64,
    bbox: Boundsf,
}

#[derive(Clone, Copy)]
pub(super) enum InsertAnchor {
    BottomLeft,
    BottomRight,
    TopLeft,
    TopRight,
}

pub(super) fn sample_insert_anchor(rng: &mut impl Random, params: &InsertParams) -> InsertAnchor {
    if rng.next_f64() >= params.anchor_randomness {
        InsertAnchor::BottomLeft
    } else {
        match rng.gen_range(0, 3) {
            0 => InsertAnchor::BottomRight,
            1 => InsertAnchor::TopLeft,
            _ => InsertAnchor::TopRight,
        }
    }
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
    bay_order: &[usize],
    candidate_top_k: usize,
    candidate_select_p: f64,
    w2: f64,
    anchor: InsertAnchor,
    rng: &mut R,
    #[cfg(feature = "profile-reconstruct")] mut profile: Option<&mut InsertGreedyProfile>,
) -> Option<ScheduledBlock> {
    fn insert_candidate_cmp(
        a: &InsertCandidate,
        b: &InsertCandidate,
        anchor: InsertAnchor,
    ) -> std::cmp::Ordering {
        let bbox_order = match anchor {
            InsertAnchor::BottomLeft => a
                .bbox
                .max_x
                .total_cmp(&b.bbox.max_x)
                .then(a.bbox.max_y.total_cmp(&b.bbox.max_y)),
            InsertAnchor::BottomRight => b
                .bbox
                .min_x
                .total_cmp(&a.bbox.min_x)
                .then(a.bbox.max_y.total_cmp(&b.bbox.max_y)),
            InsertAnchor::TopLeft => a
                .bbox
                .max_x
                .total_cmp(&b.bbox.max_x)
                .then(b.bbox.min_y.total_cmp(&a.bbox.min_y)),
            InsertAnchor::TopRight => b
                .bbox
                .min_x
                .total_cmp(&a.bbox.min_x)
                .then(b.bbox.min_y.total_cmp(&a.bbox.min_y)),
        };
        a.score_delta
            .total_cmp(&b.score_delta)
            .then(a.scheduled.entry_time.cmp(&b.scheduled.entry_time))
            .then(bbox_order)
            .then(a.scheduled.block_id.cmp(&b.scheduled.block_id))
    }

    #[cfg(feature = "profile-reconstruct")]
    let setup_start = Instant::now();
    #[cfg(feature = "profile-reconstruct")]
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }

    let block_id = original.block_id;
    let block = &problem.blocks[block_id];
    let process_t = block.processing_time;
    let min_t = block.release_time.max(min_entry_time);
    let max_t = max_entry_time;
    if min_t > max_t {
        #[cfg(feature = "profile-reconstruct")]
        if let Some(profile) = profile.as_deref_mut() {
            profile.setup += setup_start.elapsed();
        }
        return None;
    }

    let current_obj2 = if w2 == 0.0 {
        0.0
    } else {
        score_z2(loads, &pre.bay_load_scale)
    };
    let original_tardiness = (original.exit_time - block.due_date).max(0);
    let mut candidates = Vec::new();
    let mut best_score_delta = f64::INFINITY;
    #[cfg(feature = "profile-reconstruct")]
    if let Some(profile) = profile.as_deref_mut() {
        profile.setup += setup_start.elapsed();
    }

    for &bay_id in bay_order {
        #[cfg(feature = "profile-reconstruct")]
        let bay_start = Instant::now();
        #[cfg(feature = "profile-reconstruct")]
        if let Some(profile) = profile.as_deref_mut() {
            profile.bays += 1;
        }
        let delta_obj2 = if w2 == 0.0 {
            0.0
        } else {
            let mut next_loads = loads.to_vec();
            next_loads[bay_id] += block.workload as f64;
            w2 * (score_z2(&next_loads, &pre.bay_load_scale) - current_obj2)
        };
        let delta_obj23 =
            delta_obj2 + problem.weights.w3 * pre.pref_penalty[block_id][bay_id] as f64;
        let min_tardiness = min_t
            .saturating_add(process_t)
            .saturating_sub(block.due_date)
            .max(0);
        let lower_score_delta = problem.weights.w1 * min_tardiness as f64 + delta_obj23;
        if lower_score_delta > best_score_delta {
            #[cfg(feature = "profile-reconstruct")]
            if let Some(profile) = profile.as_deref_mut() {
                profile.bay_scoring += bay_start.elapsed();
            }
            continue;
        }
        #[cfg(feature = "profile-reconstruct")]
        if let Some(profile) = profile.as_deref_mut() {
            profile.bay_scoring += bay_start.elapsed();
        }

        #[cfg(feature = "profile-reconstruct")]
        let scanner_start = Instant::now();
        let scanner =
            PlacementXScanner::new(problem, pre, schedule, block_id, bay_id, min_t, max_t);
        #[cfg(feature = "profile-reconstruct")]
        if let Some(profile) = profile.as_deref_mut() {
            profile.scanner_new += scanner_start.elapsed();
        }
        let mut scanner = scanner?;
        #[cfg(feature = "profile-reconstruct")]
        if let Some(profile) = profile.as_deref_mut() {
            profile.scanners += 1;
        }

        for &orient_idx in &pre.orientation_order_by_bbox[block_id] {
            #[cfg(feature = "profile-reconstruct")]
            let orientation_start = Instant::now();
            #[cfg(feature = "profile-reconstruct")]
            if let Some(profile) = profile.as_deref_mut() {
                profile.orientations += 1;
            }
            let Some(range) = pre.collision.fit_range(bay_id, block_id, orient_idx) else {
                #[cfg(feature = "profile-reconstruct")]
                if let Some(profile) = profile.as_deref_mut() {
                    profile.orientation_prepare += orientation_start.elapsed();
                }
                continue;
            };
            let bounds = pre.orientation_bbox_bounds[block_id][orient_idx];
            let mut group_best: Option<InsertCandidate> = None;

            let mut ys: Vec<i64> = (range.min_y..=range.max_y).collect();
            rng.shuffle(&mut ys);
            #[cfg(feature = "profile-reconstruct")]
            if let Some(profile) = profile.as_deref_mut() {
                profile.orientation_prepare += orientation_start.elapsed();
                profile.y_values += ys.len() as u64;
            }
            let mut remaining_y_buffer = None;

            for y in ys {
                if remaining_y_buffer == Some(0) {
                    break;
                }

                #[cfg(feature = "profile-reconstruct")]
                let scan_start = Instant::now();
                #[cfg(feature = "profile-reconstruct")]
                if let Some(profile) = profile.as_deref_mut() {
                    profile.scan_y_calls += 1;
                }
                let valid_y = scanner.scan_y(
                    orient_idx,
                    y,
                    |scheduled| {
                        let tardiness = (scheduled.exit_time - block.due_date).max(0);
                        let score_delta = problem.weights.w1 * tardiness as f64 + delta_obj23;
                        best_score_delta = best_score_delta.min(score_delta);
                        let candidate = InsertCandidate {
                            scheduled,
                            score_delta,
                            bbox: Boundsf {
                                min_x: scheduled.x as f64 + bounds.min_x,
                                min_y: scheduled.y as f64 + bounds.min_y,
                                max_x: scheduled.x as f64 + bounds.max_x,
                                max_y: scheduled.y as f64 + bounds.max_y,
                            },
                        };
                        if group_best.as_ref().is_none_or(|best| {
                            insert_candidate_cmp(&candidate, best, anchor).is_lt()
                        }) {
                            group_best = Some(candidate);
                        }
                        tardiness <= original_tardiness
                    },
                    #[cfg(feature = "profile-reconstruct")]
                    profile
                        .as_deref_mut()
                        .map(|profile| &mut profile.placement_scan),
                );
                #[cfg(feature = "profile-reconstruct")]
                if let Some(profile) = profile.as_deref_mut() {
                    profile.scan_y += scan_start.elapsed();
                }

                match &mut remaining_y_buffer {
                    None if valid_y => remaining_y_buffer = Some(params.y_buffer),
                    Some(remaining) => *remaining -= 1,
                    None => {}
                }
            }

            if let Some(candidate) = group_best {
                candidates.push(candidate);
                #[cfg(feature = "profile-reconstruct")]
                if let Some(profile) = profile.as_deref_mut() {
                    profile.candidates += 1;
                }
            }
        }
    }

    #[cfg(feature = "profile-reconstruct")]
    let finalize_start = Instant::now();
    candidates.retain(|candidate| candidate.score_delta <= best_score_delta + 1e-9);
    if candidates.is_empty() {
        #[cfg(feature = "profile-reconstruct")]
        if let Some(profile) = profile.as_deref_mut() {
            profile.finalize += finalize_start.elapsed();
        }
        return None;
    }
    candidates.sort_by(|a, b| insert_candidate_cmp(a, b, anchor));
    candidates.truncate(candidate_top_k);
    let selected = candidates
        .iter()
        .position(|_| rng.next_f64() < candidate_select_p)
        .unwrap_or(0);
    let scheduled = candidates.swap_remove(selected).scheduled;
    #[cfg(feature = "profile-reconstruct")]
    if let Some(profile) = profile.as_deref_mut() {
        profile.finalize += finalize_start.elapsed();
        profile.successes += 1;
    }
    Some(scheduled)
}
