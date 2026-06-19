use crate::{
    precompute::{BlockPlacement, CollisionPrecompute, CollisionResult},
    util::time,
    *,
};
use std::cmp::Reverse;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
struct ActiveBlock {
    block_id: usize,
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
    exit_time: i64,
}

pub fn solve(problem: &Problem, _timelimit: f64) -> Result<Solution, String> {
    eprintln!("building collision precompute...");
    let pre = CollisionPrecompute::build(problem);
    eprintln!("elapsed: {:.4}", time::elapsed_seconds());

    let n = problem.blocks.len();
    let mut operations: BTreeMap<i64, Vec<Operation>> = BTreeMap::new();
    let mut entered = vec![false; n];
    let mut finished = 0;
    let mut active: Vec<ActiveBlock> = Vec::new();
    let mut t = 0;

    while finished < n {
        let mut i = 0;
        while i < active.len() {
            if active[i].exit_time <= t {
                let block = active.swap_remove(i);
                operations.entry(t).or_default().push(Operation {
                    op_type: "EXIT",
                    block_id: block.block_id,
                    bay_id: block.bay_id,
                    x: None,
                    y: None,
                    orient_idx: None,
                });
                finished += 1;
            } else {
                i += 1;
            }
        }

        let mut candidates: Vec<usize> = (0..n)
            .filter(|&block_id| !entered[block_id] && problem.blocks[block_id].release_time <= t)
            .collect();
        candidates.sort_by_key(|&block_id| {
            let block = &problem.blocks[block_id];
            let lateness = (t + block.processing_time - block.due_date).max(0);
            (
                Reverse(lateness),
                block.due_date,
                block.release_time,
                block_id,
            )
        });

        let mut entered_any = false;
        for block_id in candidates {
            if let Some(place) = find_bottom_left(problem, &pre, block_id, &active) {
                operations.entry(t).or_default().push(Operation {
                    op_type: "ENTRY",
                    block_id,
                    bay_id: place.bay_id,
                    x: Some(place.x),
                    y: Some(place.y),
                    orient_idx: Some(place.orient_idx),
                });
                active.push(ActiveBlock {
                    block_id,
                    bay_id: place.bay_id,
                    orient_idx: place.orient_idx,
                    x: place.x,
                    y: place.y,
                    exit_time: t + problem.blocks[block_id].processing_time,
                });
                entered[block_id] = true;
                entered_any = true;
            }
        }

        if finished == n {
            break;
        }
        if !entered_any
            && active.is_empty()
            && entered
                .iter()
                .enumerate()
                .any(|(block_id, &done)| !done && problem.blocks[block_id].release_time <= t)
        {
            return Err(format!("failed to place any released block at time {t}"));
        }

        t += 1;
    }

    Ok(Solution { operations })
}

fn find_bottom_left(
    problem: &Problem,
    pre: &CollisionPrecompute,
    block_id: usize,
    active: &[ActiveBlock],
) -> Option<Placement> {
    let mut bay_order: Vec<usize> = (0..problem.bays.len()).collect();
    bay_order.sort_by_key(|&bay_id| Reverse(problem.blocks[block_id].bay_preferences[bay_id]));

    for bay_id in bay_order {
        for orient_idx in 0..problem.blocks[block_id].shape.len() {
            let Some(range) = pre.fit_range(bay_id, block_id, orient_idx) else {
                continue;
            };
            for y in range.min_y..=range.max_y {
                for x in range.min_x..=range.max_x {
                    if can_place(pre, block_id, bay_id, orient_idx, x, y, active) {
                        return Some(Placement {
                            bay_id,
                            orient_idx,
                            x,
                            y,
                        });
                    }
                }
            }
        }
    }

    None
}

fn can_place(
    pre: &CollisionPrecompute,
    block_id: usize,
    bay_id: usize,
    orient_idx: usize,
    x: i64,
    y: i64,
    active: &[ActiveBlock],
) -> bool {
    let new_place = BlockPlacement {
        block_id,
        orient_idx,
        x,
        y,
    };

    for old in active.iter().filter(|old| old.bay_id == bay_id) {
        let old_place = BlockPlacement {
            block_id: old.block_id,
            orient_idx: old.orient_idx,
            x: old.x,
            y: old.y,
        };
        if pre.crane(new_place, old_place) != CollisionResult::Clear {
            return false;
        }
        if pre.crane(old_place, new_place) != CollisionResult::Clear {
            return false;
        }
    }

    true
}
