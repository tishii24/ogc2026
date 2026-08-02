use std::{
    fs::{File, create_dir_all},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use serde::Serialize;

use crate::{Problem, ScheduledBlock};

static OUTPUT_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn init(output_dir: &Path) -> Result<(), String> {
    create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create anneal visualize directory {}: {err}",
            output_dir.display()
        )
    })?;
    OUTPUT_DIR
        .set(output_dir.to_path_buf())
        .map_err(|_| "anneal visualizer is already initialized".to_string())
}

pub(crate) struct SnapshotMeta<'a> {
    pub(crate) worker: usize,
    pub(crate) iter: usize,
    pub(crate) elapsed: f64,
    pub(crate) reason: &'a str,
    pub(crate) neighbor: Option<&'a str>,
    pub(crate) accepted: bool,
    pub(crate) improved_current: bool,
    pub(crate) improved_best: bool,
    pub(crate) score: f64,
    pub(crate) current_score: f64,
    pub(crate) best_score: f64,
    pub(crate) delta: f64,
    pub(crate) selected_block_ids: &'a [usize],
}

#[derive(Serialize)]
struct ComponentImprovement {
    before: i64,
    after: i64,
    improvement: f64,
}

#[derive(Serialize)]
struct BlockImprovement {
    block_id: usize,
    score_improvement: f64,
    reasons: Vec<&'static str>,
    tardiness: ComponentImprovement,
    preference_penalty: ComponentImprovement,
}

#[derive(Serialize)]
struct Snapshot<'a> {
    #[serde(rename = "type")]
    snapshot_type: &'static str,
    worker: usize,
    iter: usize,
    elapsed: f64,
    reason: &'a str,
    neighbor: Option<&'a str>,
    accepted: bool,
    improved_current: bool,
    improved_best: bool,
    score: f64,
    current_score: f64,
    best_score: f64,
    delta: f64,
    changed_block_ids: Vec<usize>,
    selected_block_ids: &'a [usize],
    block_improvements: Vec<BlockImprovement>,
    score13_delta: f64,
    obj2_delta: f64,
    schedule: &'a [ScheduledBlock],
}

pub(crate) struct AnnealVisualizer {
    writer: BufWriter<File>,
}

impl AnnealVisualizer {
    pub(crate) fn new(phase: &str, worker_id: usize) -> Option<Self> {
        let output_dir = OUTPUT_DIR.get()?;
        let phase_dir = output_dir.join(phase);
        create_dir_all(&phase_dir).unwrap_or_else(|err| {
            panic!(
                "failed to create anneal visualize phase directory {}: {err}",
                phase_dir.display()
            )
        });
        let path = phase_dir.join(format!("worker_{worker_id}.jsonl"));
        let file = File::create(&path).unwrap_or_else(|err| {
            panic!(
                "failed to create anneal visualize file {}: {err}",
                path.display()
            )
        });
        Some(Self {
            writer: BufWriter::new(file),
        })
    }

    pub(crate) fn snapshot(
        &mut self,
        meta: SnapshotMeta<'_>,
        problem: &Problem,
        current: &[ScheduledBlock],
        candidate: &[ScheduledBlock],
    ) {
        let mut current_by_id = vec![None; current.len()];
        let mut candidate_by_id = vec![None; candidate.len()];
        for &block in current {
            current_by_id[block.block_id] = Some(block);
        }
        for &block in candidate {
            candidate_by_id[block.block_id] = Some(block);
        }
        let changed_block_ids = candidate
            .iter()
            .filter_map(|&block| {
                (current_by_id[block.block_id] != Some(block)).then_some(block.block_id)
            })
            .collect();

        let mut score13_delta = 0.0;
        let mut block_improvements = Vec::new();
        for &block_id in meta.selected_block_ids {
            let before = current_by_id[block_id].unwrap();
            let after = candidate_by_id[block_id].unwrap();
            let block = &problem.blocks[block_id];
            let before_tardiness = (before.exit_time - block.due_date).max(0);
            let after_tardiness = (after.exit_time - block.due_date).max(0);
            let max_preference = block.bay_preferences.iter().copied().max().unwrap_or(0);
            let before_preference = max_preference - block.bay_preferences[before.bay_id];
            let after_preference = max_preference - block.bay_preferences[after.bay_id];
            let tardiness_improvement =
                problem.weights.w1 * (before_tardiness - after_tardiness) as f64;
            let preference_improvement =
                problem.weights.w3 * (before_preference - after_preference) as f64;
            let score_improvement = tardiness_improvement + preference_improvement;
            score13_delta -= score_improvement;
            if score_improvement > 1e-9 {
                let mut reasons = Vec::new();
                if tardiness_improvement > 1e-9 {
                    reasons.push("tardiness");
                }
                if preference_improvement > 1e-9 {
                    reasons.push("preference");
                }
                block_improvements.push(BlockImprovement {
                    block_id,
                    score_improvement,
                    reasons,
                    tardiness: ComponentImprovement {
                        before: before_tardiness,
                        after: after_tardiness,
                        improvement: tardiness_improvement,
                    },
                    preference_penalty: ComponentImprovement {
                        before: before_preference,
                        after: after_preference,
                        improvement: preference_improvement,
                    },
                });
            }
        }
        block_improvements.sort_by(|a, b| b.score_improvement.total_cmp(&a.score_improvement));
        let obj2_delta = meta.delta - score13_delta;
        let snapshot = Snapshot {
            snapshot_type: "snapshot",
            worker: meta.worker,
            iter: meta.iter,
            elapsed: meta.elapsed,
            reason: meta.reason,
            neighbor: meta.neighbor,
            accepted: meta.accepted,
            improved_current: meta.improved_current,
            improved_best: meta.improved_best,
            score: meta.score,
            current_score: meta.current_score,
            best_score: meta.best_score,
            delta: meta.delta,
            changed_block_ids,
            selected_block_ids: meta.selected_block_ids,
            block_improvements,
            score13_delta,
            obj2_delta,
            schedule: candidate,
        };
        serde_json::to_writer(&mut self.writer, &snapshot)
            .expect("failed to write anneal visualize snapshot");
        self.writer
            .write_all(b"\n")
            .expect("failed to terminate anneal visualize snapshot");
    }
}
