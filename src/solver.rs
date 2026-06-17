use crate::*;

pub fn solve(problem: &Problem, _timelimit: f64) -> Result<Solution, String> {
    let mut operations: BTreeMap<i64, Vec<Operation>> = BTreeMap::new();
    let mut current_time = 0_i64;

    let mut block_order: Vec<usize> = (0..problem.blocks.len()).collect();
    block_order.sort_by_key(|&block_id| {
        let block = &problem.blocks[block_id];
        (
            block.due_date,
            block.processing_time,
            block.release_time,
            block_id,
        )
    });

    for block_id in block_order {
        let block = &problem.blocks[block_id];
        let placement = choose_placement(problem, block).ok_or_else(|| {
            format!("no feasible single-block placement found for block {block_id}")
        })?;
        let entry_time = current_time.max(block.release_time);
        let exit_time = entry_time + block.processing_time;

        push_operation(
            &mut operations,
            entry_time,
            Operation {
                op_type: "ENTRY",
                block_id,
                bay_id: placement.bay_id,
                x: Some(placement.x),
                y: Some(placement.y),
                orient_idx: Some(placement.orient_idx),
            },
        );
        push_operation(
            &mut operations,
            exit_time,
            Operation {
                op_type: "EXIT",
                block_id,
                bay_id: placement.bay_id,
                x: None,
                y: None,
                orient_idx: None,
            },
        );

        current_time = exit_time;
    }

    for ops in operations.values_mut() {
        ops.sort_by_key(|op| if op.op_type == "EXIT" { 0 } else { 1 });
    }

    Ok(Solution { operations })
}

fn push_operation(operations: &mut BTreeMap<i64, Vec<Operation>>, time: i64, op: Operation) {
    operations.entry(time).or_default().push(op);
}

fn choose_placement(problem: &Problem, block: &Block) -> Option<Placement> {
    let mut bay_order: Vec<usize> = (0..problem.bays.len()).collect();
    bay_order.sort_by_key(|&bay_id| {
        let preference = block.bay_preferences.get(bay_id).copied().unwrap_or(0);
        (-preference, bay_id)
    });

    for bay_id in bay_order {
        let bay = &problem.bays[bay_id];
        for orient_idx in 0..block.shape.len() {
            if let Some((x, y)) = fitting_position(bay, &block.shape[orient_idx]) {
                return Some(Placement {
                    bay_id,
                    orient_idx,
                    x,
                    y,
                });
            }
        }
    }

    None
}

fn fitting_position(bay: &Bay, orientation: &Orientation) -> Option<(i64, i64)> {
    let (min_x, min_y, max_x, max_y) = orientation_bbox(orientation)?;
    let x = ceil_to_i64(-min_x);
    let y = ceil_to_i64(-min_y);

    if (x as f64) + max_x <= bay.width as f64 + 1e-9
        && (y as f64) + max_y <= bay.height as f64 + 1e-9
    {
        Some((x, y))
    } else {
        None
    }
}

fn orientation_bbox(orientation: &Orientation) -> Option<(f64, f64, f64, f64)> {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut found = false;

    for layer in &orientation.layers {
        for &[x, y] in layer {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            found = true;
        }
    }

    if found {
        Some((min_x, min_y, max_x, max_y))
    } else {
        None
    }
}

fn ceil_to_i64(value: f64) -> i64 {
    value.ceil().max(0.0) as i64
}
