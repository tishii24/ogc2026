use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

const TRACE_DIR_ENV: &str = "OGC_ANNEALING_TRACE_DIR";

#[derive(Clone, Copy, Default)]
pub(crate) struct AnnealingTraceState {
    pub(crate) score: f64,
    pub(crate) z1: i64,
    pub(crate) state_hash: u64,
    pub(crate) bay_hash: u64,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct AnnealingTraceDiff {
    pub(crate) changed_blocks: usize,
    pub(crate) bay_changes: usize,
    pub(crate) orientation_changes: usize,
    pub(crate) position_changes: usize,
    pub(crate) time_changes: usize,
}

pub(crate) struct AnnealingTraceEvent<'a> {
    pub(crate) event: &'a str,
    pub(crate) neighbor: Option<&'a str>,
    pub(crate) temperature: Option<f64>,
    pub(crate) accept_threshold: Option<f64>,
    pub(crate) before: AnnealingTraceState,
    pub(crate) after: AnnealingTraceState,
    pub(crate) diff: AnnealingTraceDiff,
    pub(crate) shared_revision: Option<u64>,
}

pub(crate) struct AnnealingTraceWriter {
    phase: &'static str,
    worker_id: usize,
    writer: BufWriter<File>,
}

impl AnnealingTraceWriter {
    pub(crate) fn new(phase: &'static str, worker_id: usize) -> Option<Self> {
        let dir = PathBuf::from(env::var(TRACE_DIR_ENV).ok()?);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{phase}-worker-{worker_id}.csv"));
        let mut writer = BufWriter::new(File::create(path).unwrap());
        writeln!(
            writer,
            "elapsed,iteration,phase,worker,domain,event,neighbor,temperature,accept_threshold,before_score,after_score,before_z1,after_z1,before_hash,after_hash,before_bay_hash,after_bay_hash,changed_blocks,bay_changes,orientation_changes,position_changes,time_changes,shared_revision"
        )
        .unwrap();
        Some(Self {
            phase,
            worker_id,
            writer,
        })
    }

    pub(crate) fn write(
        &mut self,
        elapsed: f64,
        iteration: usize,
        domain: usize,
        event: AnnealingTraceEvent<'_>,
    ) {
        writeln!(
            self.writer,
            "{elapsed:.6},{iteration},{},{},{domain},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{},{},{},{},{},{},{}",
            self.phase,
            self.worker_id,
            event.event,
            event.neighbor.unwrap_or(""),
            format_optional_f64(event.temperature),
            format_optional_f64(event.accept_threshold),
            event.before.score,
            event.after.score,
            event.before.z1,
            event.after.z1,
            event.before.state_hash,
            event.after.state_hash,
            event.before.bay_hash,
            event.after.bay_hash,
            event.diff.changed_blocks,
            event.diff.bay_changes,
            event.diff.orientation_changes,
            event.diff.position_changes,
            event.diff.time_changes,
            event
                .shared_revision
                .map(|value| value.to_string())
                .unwrap_or_default(),
        )
        .unwrap();
    }
}

fn format_optional_f64(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.6}")).unwrap_or_default()
}
