use crate::ScheduledBlock;
use serde::Serialize;
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
};

#[derive(Serialize)]
struct AnnealSnapshot<'a> {
    #[serde(rename = "type")]
    event_type: &'static str,
    reason: &'static str,
    worker: usize,
    iter: usize,
    elapsed: f64,
    neighbor: Option<&'static str>,
    accepted: Option<bool>,
    improved_current: bool,
    improved_best: bool,
    score: f64,
    current_score: f64,
    best_score: f64,
    delta: Option<f64>,
    schedule: &'a [ScheduledBlock],
}

pub struct WorkerVisualizer {
    writer: BufWriter<File>,
}

impl WorkerVisualizer {
    pub fn create(dir: &Path, worker_id: usize) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("worker_{worker_id}.jsonl"));
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn write_snapshot(
        &mut self,
        reason: &'static str,
        worker: usize,
        iter: usize,
        elapsed: f64,
        neighbor: Option<&'static str>,
        accepted: Option<bool>,
        improved_current: bool,
        improved_best: bool,
        score: f64,
        current_score: f64,
        best_score: f64,
        delta: Option<f64>,
        schedule: &[ScheduledBlock],
    ) -> io::Result<()> {
        let snapshot = AnnealSnapshot {
            event_type: "snapshot",
            reason,
            worker,
            iter,
            elapsed,
            neighbor,
            accepted,
            improved_current,
            improved_best,
            score,
            current_score,
            best_score,
            delta,
            schedule,
        };
        serde_json::to_writer(&mut self.writer, &snapshot)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
