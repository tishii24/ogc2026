use crate::{
    Bay, Orientation, Problem,
    annealing::{Annealer, AnnealingAttempt, AnnealingDelegate, AnnealingState},
    log,
    params::{NeighborParams, PreoptimizeSolverParams},
    solver::{PreoptimizeState, PreoptimizedBlock, sort_default_reconstruct_order},
    util::{
        rand::{RandPcg64Mcg, Random},
        time::Timer,
    },
};
use geo::{Area, BooleanOps, Coord, LineString, MultiPolygon, Polygon};

#[derive(Clone, Copy, Debug)]
struct OrientationArea {
    union_area: f64,
    bbox_area: f64,
    perimeter: f64,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

#[derive(Debug)]
pub struct PreoptimizePrecompute {
    orientation_areas: Vec<Vec<OrientationArea>>,
    min_time: i64,
    search_horizon: i64,
}

impl PreoptimizePrecompute {
    pub fn build(problem: &Problem) -> Result<Self, String> {
        let orientation_areas = problem
            .blocks
            .iter()
            .enumerate()
            .map(|(block_id, block)| {
                block
                    .shape
                    .iter()
                    .map(orientation_area)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|err| format!("block {block_id}: {err}"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        if problem.blocks.is_empty() {
            return Ok(Self {
                orientation_areas,
                min_time: 0,
                search_horizon: 0,
            });
        }

        let min_time = problem
            .blocks
            .iter()
            .map(|block| block.release_time)
            .min()
            .unwrap();
        let max_release = problem
            .blocks
            .iter()
            .map(|block| block.release_time)
            .max()
            .unwrap();
        let total_processing = problem.blocks.iter().try_fold(0i64, |sum, block| {
            sum.checked_add(block.processing_time)
                .ok_or_else(|| "sum of processing times overflowed i64".to_string())
        })?;
        let search_horizon = max_release
            .checked_add(total_processing)
            .ok_or_else(|| "preoptimize search horizon overflowed i64".to_string())?;

        Ok(Self {
            orientation_areas,
            min_time,
            search_horizon,
        })
    }
}

struct PreoptimizeContext<'a> {
    pre: &'a PreoptimizePrecompute,
    params: &'a PreoptimizeSolverParams,
    reconstruct_order_params: &'a NeighborParams,
    occupancy: Vec<f64>,
    block_areas: Vec<f64>,
    capacity: f64,
    max_time: i64,
}

#[derive(Clone)]
struct PreoptimizeAnnealingState {
    schedule: Vec<PreoptimizedBlock>,
    used_area: Vec<f64>,
    z1: f64,
    congestion: f64,
    objective: f64,
}

impl AnnealingState for PreoptimizeAnnealingState {
    fn annealing_score(&self) -> f64 {
        self.objective
    }

    fn tabu_key(&self) -> Option<u64> {
        None
    }
}

fn evaluate_schedule(problem: &Problem, schedule: &[PreoptimizedBlock]) -> f64 {
    schedule
        .iter()
        .enumerate()
        .map(|(block_id, selected)| block_z1(problem, block_id, *selected))
        .sum()
}

fn layer_polygon(layer: &[[f64; 2]]) -> Polygon<f64> {
    let mut coords: Vec<_> = layer.iter().map(|&[x, y]| Coord { x, y }).collect();
    if coords.first() != coords.last() {
        coords.push(coords[0]);
    }
    Polygon::new(LineString::from(coords), vec![])
}

fn orientation_area(orientation: &Orientation) -> Result<OrientationArea, String> {
    let mut polygons = orientation.layers.iter().map(|layer| layer_polygon(layer));
    let first = polygons
        .next()
        .ok_or_else(|| "orientation must contain at least one layer".to_string())?;
    let mut union = MultiPolygon(vec![first]);
    for polygon in polygons {
        union = union.union(&polygon);
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for layer in &orientation.layers {
        for &[x, y] in layer {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    let perimeter = union
        .0
        .iter()
        .flat_map(|polygon| std::iter::once(polygon.exterior()).chain(polygon.interiors().iter()))
        .flat_map(|ring| ring.0.windows(2))
        .map(|edge| {
            let dx = edge[1].x - edge[0].x;
            let dy = edge[1].y - edge[0].y;
            dx.hypot(dy)
        })
        .sum();

    Ok(OrientationArea {
        union_area: union.unsigned_area(),
        bbox_area: (max_x - min_x) * (max_y - min_y),
        perimeter,
        min_x,
        min_y,
        max_x,
        max_y,
    })
}

fn fits_bay(area: OrientationArea, bay: &Bay) -> bool {
    let min_integer_x = (-area.min_x).ceil() as i64;
    let max_integer_x = (bay.width as f64 - area.max_x).floor() as i64;
    let min_integer_y = (-area.min_y).ceil() as i64;
    let max_integer_y = (bay.height as f64 - area.max_y).floor() as i64;
    min_integer_x <= max_integer_x && min_integer_y <= max_integer_y
}

fn build_occupancy(
    problem: &Problem,
    pre: &PreoptimizePrecompute,
    params: &PreoptimizeSolverParams,
) -> Result<Vec<f64>, String> {
    pre.orientation_areas
        .iter()
        .enumerate()
        .map(|(block_id, areas)| {
            areas
                .iter()
                .copied()
                .filter(|&area| problem.bays.iter().any(|bay| fits_bay(area, bay)))
                .map(|area| {
                    area.union_area
                        + params.alpha * (area.bbox_area - area.union_area)
                        + params.beta * area.perimeter
                })
                .min_by(f64::total_cmp)
                .ok_or_else(|| format!("block {block_id} has no eligible bay"))
        })
        .collect()
}

fn build_initial_state(
    problem: &Problem,
    context: &mut PreoptimizeContext<'_>,
) -> Result<PreoptimizeAnnealingState, String> {
    let time_count: usize = (context.pre.search_horizon - context.pre.min_time)
        .try_into()
        .map_err(|_| "greedy time range is too large".to_string())?;
    let mut used_area = vec![0.0; time_count];
    let mut congestion = 0.0;
    let mut schedule = vec![None; problem.blocks.len()];
    let mut order: Vec<usize> = (0..problem.blocks.len()).collect();
    order.sort_by(|&a, &b| {
        let block_a = &problem.blocks[a];
        let block_b = &problem.blocks[b];
        let slack_a = block_a.due_date - block_a.release_time - block_a.processing_time;
        let slack_b = block_b.due_date - block_b.release_time - block_b.processing_time;
        slack_a
            .cmp(&slack_b)
            .then(block_a.due_date.cmp(&block_b.due_date))
            .then(context.occupancy[b].total_cmp(&context.occupancy[a]))
            .then(a.cmp(&b))
    });

    for block_id in order {
        let block = &problem.blocks[block_id];
        let occupied_area = context.occupancy[block_id];
        let occupied_duration = block.processing_time;
        let last_entry = context.pre.search_horizon - occupied_duration;
        let mut best: Option<(f64, i64)> = None;
        for entry_time in block.release_time..=last_entry {
            if !(entry_time..entry_time + occupied_duration).all(|time| {
                used_area[(time - context.pre.min_time) as usize] + occupied_area
                    <= context.capacity + 1e-9
            }) {
                continue;
            }

            let tardiness = (entry_time + block.processing_time - block.due_date).max(0);
            let ratio = occupied_area / context.capacity;
            let congestion_delta = (entry_time..entry_time + occupied_duration)
                .map(|time| {
                    ratio * used_area[(time - context.pre.min_time) as usize] / context.capacity
                })
                .sum::<f64>();
            let score = problem.weights.w1 * tardiness as f64
                + problem.weights.w1 * context.params.congestion_weight * congestion_delta;
            let candidate = (score, entry_time);
            if best.as_ref().is_none_or(|best| {
                candidate
                    .0
                    .total_cmp(&best.0)
                    .then(candidate.1.cmp(&best.1))
                    .is_lt()
            }) {
                best = Some(candidate);
            }
        }

        let (_, entry_time) =
            best.ok_or_else(|| format!("failed to greedily schedule block {block_id}"))?;
        let selected = PreoptimizedBlock { entry_time };
        add_used_area(
            problem,
            context,
            &mut used_area,
            &mut congestion,
            block_id,
            selected,
            1.0,
        );
        schedule[block_id] = Some(selected);
    }

    let schedule: Vec<_> = schedule
        .into_iter()
        .enumerate()
        .map(|(block_id, selected)| {
            selected.ok_or_else(|| format!("block {block_id} was not greedily scheduled"))
        })
        .collect::<Result<_, _>>()?;
    let initial_max_end = schedule
        .iter()
        .enumerate()
        .map(|(block_id, selected)| selected.entry_time + problem.blocks[block_id].processing_time)
        .max()
        .unwrap();
    let max_due = problem
        .blocks
        .iter()
        .map(|block| block.due_date)
        .max()
        .unwrap();
    context.max_time = initial_max_end.max(max_due.min(context.pre.search_horizon));
    used_area.truncate((context.max_time - context.pre.min_time) as usize);

    let z1 = evaluate_schedule(problem, &schedule);
    Ok(PreoptimizeAnnealingState {
        schedule,
        used_area,
        z1,
        congestion,
        objective: problem.weights.w1 * z1
            + problem.weights.w1 * context.params.congestion_weight * congestion,
    })
}

fn block_z1(problem: &Problem, block_id: usize, selected: PreoptimizedBlock) -> f64 {
    let block = &problem.blocks[block_id];
    (selected.entry_time + block.processing_time - block.due_date).max(0) as f64
}

fn add_used_area(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    used_area: &mut [f64],
    congestion: &mut f64,
    block_id: usize,
    selected: PreoptimizedBlock,
    sign: f64,
) {
    let area = context.occupancy[block_id];
    let ratio = area / context.capacity;
    let block = &problem.blocks[block_id];
    let end_time = selected.entry_time + block.processing_time;
    for time in selected.entry_time..end_time {
        let used = &mut used_area[(time - context.pre.min_time) as usize];
        let used_ratio = *used / context.capacity;
        if sign > 0.0 {
            *congestion += ratio * used_ratio;
        } else {
            *congestion -= ratio * (used_ratio - ratio);
        }
        *used += sign * area;
    }
}

fn can_place(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    used_area: &[f64],
    block_id: usize,
    selected: PreoptimizedBlock,
) -> bool {
    let block = &problem.blocks[block_id];
    let end_time = selected.entry_time + block.processing_time;
    if selected.entry_time < block.release_time || end_time > context.max_time {
        return false;
    }
    let area = context.occupancy[block_id];
    (selected.entry_time..end_time).all(|time| {
        used_area[(time - context.pre.min_time) as usize] + area <= context.capacity + 1e-9
    })
}

fn select_block(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> usize {
    if rng.nextf() >= context.params.neighbor.bad_block_select_probability {
        return rng.gen_index(problem.blocks.len());
    }
    let mut best = rng.gen_index(problem.blocks.len());
    let mut best_cost = f64::NEG_INFINITY;
    for _ in 0..context.params.neighbor.bad_block_sample_count {
        let block_id = rng.gen_index(problem.blocks.len());
        let selected = state.schedule[block_id];
        let cost = problem.weights.w1 * block_z1(problem, block_id, selected);
        if cost > best_cost {
            best = block_id;
            best_cost = cost;
        }
    }
    best
}

fn sample_entry_time(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    block_id: usize,
    current: i64,
    rng: &mut impl Random,
) -> i64 {
    let block = &problem.blocks[block_id];
    let min_time = block.release_time;
    let max_time = context.max_time - block.processing_time;
    match rng.gen_range(0, 4) {
        0 => min_time,
        1 => (block.due_date - block.processing_time).clamp(min_time, max_time),
        2 => {
            let max_shift = context.params.neighbor.max_time_shift;
            let delta = rng.gen_range(0, (2 * max_shift + 1) as usize) as i64 - max_shift;
            current.saturating_add(delta).clamp(min_time, max_time)
        }
        _ => min_time + rng.gen_range(0, (max_time - min_time + 1) as usize) as i64,
    }
}

fn try_relocate(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> bool {
    let block_id = select_block(problem, context, state, rng);
    let old = state.schedule[block_id];
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        block_id,
        old,
        -1.0,
    );

    let mut candidate = None;
    for _ in 0..context.params.neighbor.max_relocate_attempts {
        let selected = PreoptimizedBlock {
            entry_time: sample_entry_time(problem, context, block_id, old.entry_time, rng),
        };
        if selected != old && can_place(problem, context, &state.used_area, block_id, selected) {
            candidate = Some(selected);
            break;
        }
    }
    let Some(selected) = candidate else {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            block_id,
            old,
            1.0,
        );
        return false;
    };

    let old_z1 = block_z1(problem, block_id, old);
    let new_z1 = block_z1(problem, block_id, selected);
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        block_id,
        selected,
        1.0,
    );
    state.schedule[block_id] = selected;
    state.z1 += new_z1 - old_z1;
    state.objective = problem.weights.w1 * state.z1
        + problem.weights.w1 * context.params.congestion_weight * state.congestion;
    true
}

fn try_swap(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> bool {
    if problem.blocks.len() < 2 {
        return false;
    }
    let a = select_block(problem, context, state, rng);
    let mut b = rng.gen_index(problem.blocks.len() - 1);
    if b >= a {
        b += 1;
    }
    let old_a = state.schedule[a];
    let old_b = state.schedule[b];
    let new_a = PreoptimizedBlock {
        entry_time: old_b.entry_time,
    };
    let new_b = PreoptimizedBlock {
        entry_time: old_a.entry_time,
    };
    if new_a == old_a && new_b == old_b {
        return false;
    }

    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        a,
        old_a,
        -1.0,
    );
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        b,
        old_b,
        -1.0,
    );
    if !can_place(problem, context, &state.used_area, a, new_a) {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            a,
            old_a,
            1.0,
        );
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            b,
            old_b,
            1.0,
        );
        return false;
    }
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        a,
        new_a,
        1.0,
    );
    if !can_place(problem, context, &state.used_area, b, new_b) {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            a,
            new_a,
            -1.0,
        );
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            a,
            old_a,
            1.0,
        );
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            b,
            old_b,
            1.0,
        );
        return false;
    }
    add_used_area(
        problem,
        context,
        &mut state.used_area,
        &mut state.congestion,
        b,
        new_b,
        1.0,
    );

    let old_z1 = block_z1(problem, a, old_a) + block_z1(problem, b, old_b);
    let new_z1 = block_z1(problem, a, new_a) + block_z1(problem, b, new_b);
    state.schedule[a] = new_a;
    state.schedule[b] = new_b;
    state.z1 += new_z1 - old_z1;
    state.objective = problem.weights.w1 * state.z1
        + problem.weights.w1 * context.params.congestion_weight * state.congestion;
    true
}

fn sample_large_reconstruct_count(
    context: &PreoptimizeContext<'_>,
    rng: &mut impl Random,
) -> usize {
    let params = &context.params.neighbor;
    let span = params.max_removed_blocks - params.min_removed_blocks + 1;
    let u = rng.nextf().powf(params.remove_count_sample_power);
    params.min_removed_blocks + ((u * span as f64) as usize).min(span - 1)
}

fn choose_large_reconstruct_blocks(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &PreoptimizeAnnealingState,
    k: usize,
    rng: &mut impl Random,
) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    let seed = select_block(problem, context, state, rng);
    let mut remaining: Vec<_> = (0..problem.blocks.len())
        .filter(|&block_id| block_id != seed)
        .collect();
    rng.shuffle(&mut remaining);
    let mut selected = Vec::with_capacity(k);
    selected.push(seed);
    selected.extend(remaining.into_iter().take(k - 1));
    selected
}

fn try_large_reconstruct(
    problem: &Problem,
    context: &PreoptimizeContext<'_>,
    state: &mut PreoptimizeAnnealingState,
    rng: &mut impl Random,
) -> bool {
    let k = sample_large_reconstruct_count(context, rng).min(problem.blocks.len());
    if k < 2 {
        return false;
    }
    let mut removed_ids = choose_large_reconstruct_blocks(problem, context, state, k, rng);
    for &block_id in &removed_ids {
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            block_id,
            state.schedule[block_id],
            -1.0,
        );
    }

    sort_default_reconstruct_order(
        problem,
        &context.block_areas,
        &mut removed_ids,
        rng,
        context.reconstruct_order_params,
    );
    let mut changed = false;
    for block_id in removed_ids {
        let old = state.schedule[block_id];
        let mut candidate = None;
        for _ in 0..context.params.neighbor.max_relocate_attempts {
            let selected = PreoptimizedBlock {
                entry_time: sample_entry_time(problem, context, block_id, old.entry_time, rng),
            };
            if can_place(problem, context, &state.used_area, block_id, selected) {
                candidate = Some(selected);
                break;
            }
        }
        let Some(selected) = candidate else {
            return false;
        };
        add_used_area(
            problem,
            context,
            &mut state.used_area,
            &mut state.congestion,
            block_id,
            selected,
            1.0,
        );
        state.schedule[block_id] = selected;
        changed |= selected != old;
    }
    if !changed {
        return false;
    }

    state.z1 = evaluate_schedule(problem, &state.schedule);
    state.objective = problem.weights.w1 * state.z1
        + problem.weights.w1 * context.params.congestion_weight * state.congestion;
    true
}

const PREOPTIMIZE_NEIGHBOR_KINDS: &[&str] = &["Relocate", "Swap", "LargeReconstruct"];

struct PreoptimizeAnnealingDelegate<'a> {
    problem: &'a Problem,
    context: PreoptimizeContext<'a>,
    initial: PreoptimizeAnnealingState,
}

impl AnnealingDelegate for PreoptimizeAnnealingDelegate<'_> {
    type State = PreoptimizeAnnealingState;
    type Output = PreoptimizeState;

    fn initial_states(&self) -> Vec<Self::State> {
        vec![self.initial.clone()]
    }

    fn name(&self) -> &'static str {
        "preopt"
    }

    fn neighbor_kinds(&self) -> &'static [&'static str] {
        PREOPTIMIZE_NEIGHBOR_KINDS
    }

    fn exchange_threshold(&self, _domain: usize) -> f64 {
        self.context.params.exchange_threshold_w1_scale * self.problem.weights.w1
    }

    fn propose(
        &self,
        _domain: usize,
        current: &Self::State,
        _accept_threshold: f64,
        rng: &mut RandPcg64Mcg,
    ) -> AnnealingAttempt<Self::State> {
        let mut candidate = current.clone();
        let probabilities = &self.context.params.neighbor_probabilities;
        let total = probabilities.relocate + probabilities.swap + probabilities.large_reconstruct;
        let x = rng.nextf() * total;
        let (neighbor_kind, succeeded) = if x < probabilities.large_reconstruct {
            (
                2,
                try_large_reconstruct(self.problem, &self.context, &mut candidate, rng),
            )
        } else if x < probabilities.large_reconstruct + probabilities.swap {
            (
                1,
                try_swap(self.problem, &self.context, &mut candidate, rng),
            )
        } else {
            (
                0,
                try_relocate(self.problem, &self.context, &mut candidate, rng),
            )
        };
        AnnealingAttempt {
            neighbor_kind,
            candidate: succeeded.then_some(candidate),
        }
    }

    fn is_finished(&self, _domain: usize, _state: &Self::State) -> bool {
        false
    }

    fn finish(&self, mut states: Vec<Self::State>) -> Self::Output {
        let state = states.pop().unwrap();
        let tardiness = evaluate_schedule(self.problem, &state.schedule);
        let score = self.problem.weights.w1 * tardiness;
        log!(
            "[preopt] objective: tardiness={tardiness:.3}, congestion={:.3}, search={:.3}",
            state.congestion,
            state.objective,
        );
        PreoptimizeState {
            score,
            blocks: state.schedule,
        }
    }
}

pub fn preoptimize(
    problem: &Problem,
    pre: &PreoptimizePrecompute,
    params: &PreoptimizeSolverParams,
    reconstruct_order_params: &NeighborParams,
    time_limit: f64,
    max_worker_count: usize,
    seed: u64,
) -> Result<PreoptimizeState, String> {
    let mut capacity = 0.0;
    for bay in problem.bays.iter() {
        let width = (bay.width as f64 - params.bay_padding).max(1.0);
        let height = (bay.height as f64 - params.bay_padding).max(1.0);
        capacity += width * height;
    }

    let occupancy = build_occupancy(problem, pre, params)?;
    if problem.blocks.is_empty() {
        log!("[preopt] objective: tardiness=0.000, congestion=0.000, search=0.000");
        return Ok(PreoptimizeState {
            score: 0.0,
            blocks: Vec::new(),
        });
    }

    let mut context = PreoptimizeContext {
        pre,
        params,
        reconstruct_order_params,
        block_areas: occupancy.clone(),
        occupancy,
        capacity,
        max_time: pre.search_horizon,
    };
    let initial = build_initial_state(problem, &mut context)?;
    let annealing_params = params.annealing.make(problem, initial.annealing_score());
    let timer = Timer::start(1.0);
    let worker_count = rayon::current_num_threads().clamp(1, max_worker_count);
    let delegate = PreoptimizeAnnealingDelegate {
        problem,
        context,
        initial,
    };
    Ok(Annealer::new(time_limit, worker_count, seed, annealing_params, delegate).run(timer))
}
