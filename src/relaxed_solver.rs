#![allow(dead_code)]

use crate::{util::time, *};
use std::cmp::Ordering;

const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
pub struct PresolveBlock {
    pub block_id: usize,
    pub bay_id: usize,
    pub entry_time: i64,
    pub exit_time: i64,
}

#[derive(Clone, Debug)]
pub struct PresolveResult {
    pub score: f64,
    pub obj1: i64,
    pub obj2: f64,
    pub obj3: i64,
    pub schedule: Vec<PresolveBlock>,
}

struct PresolvePrecompute {
    bay_area: Vec<f64>,
    bay_load_scale: Vec<f64>,
    block_area: Vec<f64>,
    pref_penalty: Vec<Vec<i64>>,
    bay_order_by_pref: Vec<Vec<usize>>,
    horizon: i64,
}

pub fn solve_relaxed_score(problem: &Problem, timelimit: f64) -> Result<PresolveResult, String> {
    let pre = PresolvePrecompute::build(problem);
    let deadline = time::elapsed_seconds() + timelimit.max(0.0);

    let mut schedule = build_initial_schedule(problem, &pre)?;
    let mut usage = build_usage(problem, &pre, &schedule);
    let mut current = evaluate(problem, &pre, &schedule);

    while time::elapsed_seconds() < deadline {
        let mut improved = false;
        let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
        order.sort_by(|&a, &b| {
            block_badness(problem, &pre, schedule[b])
                .cmp(&block_badness(problem, &pre, schedule[a]))
                .then_with(|| a.cmp(&b))
        });

        for block_id in order {
            if time::elapsed_seconds() >= deadline {
                break;
            }

            let old = schedule[block_id];
            remove_usage(&mut usage, old, pre.block_area[block_id]);

            let mut best_block = old;
            let mut best_score = current.score;
            for bay_id in 0..problem.bays.len() {
                if pre.block_area[block_id] > pre.bay_area[bay_id] + EPS {
                    continue;
                }
                for entry_time in problem.blocks[block_id].release_time..=pre.horizon {
                    let candidate = make_presolve_block(problem, block_id, bay_id, entry_time);
                    if candidate.exit_time > pre.horizon + 1 {
                        break;
                    }
                    if !can_place(&usage, &pre, candidate) {
                        continue;
                    }
                    let score = evaluate_move(problem, &pre, &schedule, block_id, candidate).score;
                    if score + EPS < best_score
                        || ((score - best_score).abs() <= EPS
                            && placement_order(candidate, best_block) == Ordering::Less)
                    {
                        best_score = score;
                        best_block = candidate;
                    }
                }
            }

            schedule[block_id] = best_block;
            add_usage(&mut usage, best_block, pre.block_area[block_id]);
            if best_score + EPS < current.score {
                current = evaluate(problem, &pre, &schedule);
                improved = true;
            } else {
                schedule[block_id] = old;
                remove_usage(&mut usage, best_block, pre.block_area[block_id]);
                add_usage(&mut usage, old, pre.block_area[block_id]);
            }
        }

        if !improved {
            break;
        }
    }

    Ok(evaluate(problem, &pre, &schedule))
}

impl PresolvePrecompute {
    fn build(problem: &Problem) -> Self {
        let bay_area: Vec<f64> = problem
            .bays
            .iter()
            .map(|bay| (bay.width * bay.height) as f64)
            .collect();
        let avg_area = if bay_area.is_empty() {
            0.0
        } else {
            bay_area.iter().sum::<f64>() / bay_area.len() as f64
        };
        let bay_load_scale = bay_area
            .iter()
            .map(|&area| if area > 0.0 { avg_area / area } else { 0.0 })
            .collect();

        let block_area = problem.blocks.iter().map(relaxed_block_area).collect();
        let pref_penalty = problem
            .blocks
            .iter()
            .map(|block| {
                let max_pref = block.bay_preferences.iter().copied().max().unwrap_or(0);
                block
                    .bay_preferences
                    .iter()
                    .map(|&pref| max_pref - pref)
                    .collect()
            })
            .collect();
        let bay_order_by_pref = problem
            .blocks
            .iter()
            .map(|block| {
                let mut order: Vec<usize> = (0..problem.bays.len()).collect();
                order.sort_by_key(|&bay_id| {
                    (
                        std::cmp::Reverse(block.bay_preferences.get(bay_id).copied().unwrap_or(0)),
                        bay_id,
                    )
                });
                order
            })
            .collect();
        let max_release = problem
            .blocks
            .iter()
            .map(|block| block.release_time)
            .max()
            .unwrap_or(0);
        let max_due = problem
            .blocks
            .iter()
            .map(|block| block.due_date)
            .max()
            .unwrap_or(0);
        let total_processing: i64 = problem
            .blocks
            .iter()
            .map(|block| block.processing_time)
            .sum();
        let horizon = max_release.max(max_due) + total_processing + 1;

        Self {
            bay_area,
            bay_load_scale,
            block_area,
            pref_penalty,
            bay_order_by_pref,
            horizon,
        }
    }
}

fn build_initial_schedule(
    problem: &Problem,
    pre: &PresolvePrecompute,
) -> Result<Vec<PresolveBlock>, String> {
    let mut schedule: Vec<_> = (0..problem.blocks.len())
        .map(|block_id| {
            make_presolve_block(problem, block_id, 0, problem.blocks[block_id].release_time)
        })
        .collect();
    let mut placed = vec![false; problem.blocks.len()];
    let mut usage = vec![vec![0.0; (pre.horizon + 1).max(0) as usize]; problem.bays.len()];
    let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
    order.sort_by_key(|&block_id| {
        let block = &problem.blocks[block_id];
        (block.due_date, block.release_time, block_id)
    });

    for block_id in order {
        let mut best = None;
        let mut best_result = None;
        for &bay_id in &pre.bay_order_by_pref[block_id] {
            if pre.block_area[block_id] > pre.bay_area[bay_id] + EPS {
                continue;
            }
            if let Some(entry_time) = earliest_entry(problem, pre, &usage, block_id, bay_id) {
                let candidate = make_presolve_block(problem, block_id, bay_id, entry_time);
                let result =
                    evaluate_initial_candidate(problem, pre, &schedule, &placed, candidate);
                if best_result
                    .as_ref()
                    .is_none_or(|old: &PresolveResult| result.score + EPS < old.score)
                {
                    best = Some(candidate);
                    best_result = Some(result);
                }
            }
        }

        let Some(block) = best else {
            return Err(format!(
                "failed to place block {block_id} in relaxed presolver"
            ));
        };
        schedule[block_id] = block;
        placed[block_id] = true;
        add_usage(&mut usage, block, pre.block_area[block_id]);
    }

    Ok(schedule)
}

fn earliest_entry(
    problem: &Problem,
    pre: &PresolvePrecompute,
    usage: &[Vec<f64>],
    block_id: usize,
    bay_id: usize,
) -> Option<i64> {
    for entry_time in problem.blocks[block_id].release_time..=pre.horizon {
        let candidate = make_presolve_block(problem, block_id, bay_id, entry_time);
        if candidate.exit_time > pre.horizon + 1 {
            return None;
        }
        if can_place(usage, pre, candidate) {
            return Some(entry_time);
        }
    }
    None
}

fn build_usage(
    problem: &Problem,
    pre: &PresolvePrecompute,
    schedule: &[PresolveBlock],
) -> Vec<Vec<f64>> {
    let mut usage = vec![vec![0.0; (pre.horizon + 1).max(0) as usize]; problem.bays.len()];
    for &block in schedule {
        add_usage(&mut usage, block, pre.block_area[block.block_id]);
    }
    usage
}

fn can_place(usage: &[Vec<f64>], pre: &PresolvePrecompute, block: PresolveBlock) -> bool {
    let area = pre.block_area[block.block_id];
    if area > pre.bay_area[block.bay_id] + EPS {
        return false;
    }
    let Some(row) = usage.get(block.bay_id) else {
        return false;
    };
    for t in block.entry_time..block.exit_time {
        let t = t as usize;
        if t >= row.len() || row[t] + area > pre.bay_area[block.bay_id] + EPS {
            return false;
        }
    }
    true
}

fn add_usage(usage: &mut [Vec<f64>], block: PresolveBlock, area: f64) {
    for t in block.entry_time..block.exit_time {
        usage[block.bay_id][t as usize] += area;
    }
}

fn remove_usage(usage: &mut [Vec<f64>], block: PresolveBlock, area: f64) {
    for t in block.entry_time..block.exit_time {
        usage[block.bay_id][t as usize] -= area;
    }
}

fn evaluate_initial_candidate(
    problem: &Problem,
    pre: &PresolvePrecompute,
    schedule: &[PresolveBlock],
    placed: &[bool],
    candidate: PresolveBlock,
) -> PresolveResult {
    let mut blocks = Vec::new();
    for (&block, &is_placed) in schedule.iter().zip(placed.iter()) {
        if is_placed {
            blocks.push(block);
        }
    }
    blocks.push(candidate);
    evaluate(problem, pre, &blocks)
}

fn evaluate_move(
    problem: &Problem,
    pre: &PresolvePrecompute,
    schedule: &[PresolveBlock],
    block_id: usize,
    candidate: PresolveBlock,
) -> PresolveResult {
    let mut next = schedule.to_vec();
    next[block_id] = candidate;
    evaluate(problem, pre, &next)
}

fn evaluate(
    problem: &Problem,
    pre: &PresolvePrecompute,
    schedule: &[PresolveBlock],
) -> PresolveResult {
    evaluate_partial(problem, pre, schedule, schedule.len())
}

fn evaluate_partial(
    problem: &Problem,
    pre: &PresolvePrecompute,
    schedule: &[PresolveBlock],
    len: usize,
) -> PresolveResult {
    let mut obj1 = 0i64;
    let mut obj3 = 0i64;
    let mut loads = vec![0.0; problem.bays.len()];
    let mut result_schedule = Vec::with_capacity(len);

    for &s in schedule.iter().take(len) {
        let block = &problem.blocks[s.block_id];
        obj1 += (s.exit_time - block.due_date).max(0);
        obj3 += pre.pref_penalty[s.block_id][s.bay_id];
        loads[s.bay_id] += block.workload as f64;
        result_schedule.push(s);
    }

    let obj2 = normalized_imbalance(pre, &loads);
    let score = problem.weights.w1 * obj1 as f64
        + problem.weights.w2 * obj2
        + problem.weights.w3 * obj3 as f64;

    PresolveResult {
        score,
        obj1,
        obj2,
        obj3,
        schedule: result_schedule,
    }
}

fn normalized_imbalance(pre: &PresolvePrecompute, loads: &[f64]) -> f64 {
    if loads.len() < 2 {
        return 0.0;
    }
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    for (bay_id, &load) in loads.iter().enumerate() {
        let normalized = load * pre.bay_load_scale[bay_id];
        min_value = min_value.min(normalized);
        max_value = max_value.max(normalized);
    }
    (max_value - min_value).floor()
}

fn block_badness(problem: &Problem, pre: &PresolvePrecompute, block: PresolveBlock) -> i64 {
    let p = &problem.blocks[block.block_id];
    (block.exit_time - p.due_date).max(0) * 10_000
        + pre.pref_penalty[block.block_id][block.bay_id] * 100
}

fn make_presolve_block(
    problem: &Problem,
    block_id: usize,
    bay_id: usize,
    entry_time: i64,
) -> PresolveBlock {
    PresolveBlock {
        block_id,
        bay_id,
        entry_time,
        exit_time: entry_time + problem.blocks[block_id].processing_time,
    }
}

fn placement_order(a: PresolveBlock, b: PresolveBlock) -> Ordering {
    (a.exit_time, a.entry_time, a.bay_id, a.block_id).cmp(&(
        b.exit_time,
        b.entry_time,
        b.bay_id,
        b.block_id,
    ))
}

fn relaxed_block_area(block: &Block) -> f64 {
    block
        .shape
        .iter()
        .map(|orient| {
            orient
                .layers
                .iter()
                .map(|layer| polygon_area(layer))
                .fold(0.0, f64::max)
        })
        .fold(f64::INFINITY, f64::min)
}

fn polygon_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        area += points[i][0] * points[j][1] - points[j][0] * points[i][1];
    }
    area.abs() * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_schedule_respects_area_capacity() {
        let problem = Problem {
            bays: vec![Bay {
                width: 10,
                height: 10,
            }],
            blocks: vec![block(0, 100, 3, 60.0), block(0, 100, 3, 50.0)],
            weights: Weights::default(),
        };

        let result = solve_relaxed_score(&problem, 0.0).unwrap();
        assert_eq!(result.schedule.len(), 2);
        assert!(
            result.schedule[0].exit_time <= result.schedule[1].entry_time
                || result.schedule[1].exit_time <= result.schedule[0].entry_time
        );
    }

    fn block(release_time: i64, due_date: i64, processing_time: i64, area: f64) -> Block {
        Block {
            release_time,
            due_date,
            processing_time,
            workload: 1,
            bay_preferences: vec![100],
            shape: vec![Orientation {
                layers: vec![rectangle(area, 1.0)],
            }],
        }
    }

    fn rectangle(width: f64, height: f64) -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]
    }
}
