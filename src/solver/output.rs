use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::Mutex,
};

use crate::{Operation, ScheduledBlock, Solution, utils::time::Timer};

use super::optimize::OptimizeState;

pub(super) fn schedule_to_solution(schedule: &[ScheduledBlock]) -> Solution {
    let mut operations: BTreeMap<i64, Vec<Operation>> = BTreeMap::new();

    for scheduled in schedule {
        operations
            .entry(scheduled.exit_time)
            .or_default()
            .push(Operation {
                op_type: "EXIT",
                block_id: scheduled.block_id,
                bay_id: scheduled.bay_id,
                x: None,
                y: None,
                orient_idx: None,
            });
    }
    for scheduled in schedule {
        operations
            .entry(scheduled.entry_time)
            .or_default()
            .push(Operation {
                op_type: "ENTRY",
                block_id: scheduled.block_id,
                bay_id: scheduled.bay_id,
                x: Some(scheduled.x),
                y: Some(scheduled.y),
                orient_idx: Some(scheduled.orient_idx),
            });
    }

    Solution { operations }
}

#[derive(serde::Serialize)]
struct SolutionCandidate {
    score: f64,
    solution: Solution,
}

pub(super) struct CandidateEmitter {
    interval_seconds: f64,
    last_emitted: Mutex<f64>,
}

impl CandidateEmitter {
    pub(super) fn new(interval_seconds: f64) -> Self {
        Self {
            interval_seconds,
            last_emitted: Mutex::new(f64::NEG_INFINITY),
        }
    }

    pub(super) fn emit(&self, state: &OptimizeState, timer: Timer, force: bool) {
        let elapsed = timer.elapsed_seconds();
        let mut last_emitted = self.last_emitted.lock().unwrap();
        if !force && elapsed - *last_emitted < self.interval_seconds {
            return;
        }

        let emit_started = timer.elapsed_seconds();
        let serialize_started = timer.elapsed_seconds();
        let candidate = SolutionCandidate {
            score: state.objective,
            solution: schedule_to_solution(&state.schedule),
        };
        let output = match serde_json::to_string(&candidate) {
            Ok(output) => output,
            Err(err) => {
                log!(
                    "[{:.4}] [candidate-emit] score={:.3}, force={}, status=serialize-failed, error={}, total={:.4}s",
                    timer.elapsed_seconds(),
                    state.objective,
                    force,
                    err,
                    timer.elapsed_seconds() - emit_started,
                );
                return;
            }
        };
        let serialize_seconds = timer.elapsed_seconds() - serialize_started;

        let write_started = timer.elapsed_seconds();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();
        let succeeded = writeln!(stdout, "{output}").is_ok() && stdout.flush().is_ok();
        let write_seconds = timer.elapsed_seconds() - write_started;
        let total_seconds = timer.elapsed_seconds() - emit_started;
        if succeeded {
            *last_emitted = elapsed;
        }
        log!(
            "[{:.4}] [candidate-emit] score={:.3}, force={}, status={}, bytes={}, serialize={:.6}s, write={:.6}s, total={:.6}s",
            timer.elapsed_seconds(),
            state.objective,
            force,
            if succeeded { "ok" } else { "write-failed" },
            output.len(),
            serialize_seconds,
            write_seconds,
            total_seconds,
        );
    }
}
