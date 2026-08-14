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

struct EmitState {
    interval_seconds: f64,
    last_emitted: f64,
}

pub(super) struct CandidateEmitter {
    min_interval_seconds: f64,
    max_count: usize,
    state: Mutex<EmitState>,
}

impl CandidateEmitter {
    pub(super) fn new(min_interval_seconds: f64, max_count: usize) -> Self {
        Self {
            min_interval_seconds,
            max_count,
            state: Mutex::new(EmitState {
                interval_seconds: f64::INFINITY,
                last_emitted: f64::NEG_INFINITY,
            }),
        }
    }

    pub(super) fn configure(&self, final_horizon_time: f64, _timer: Timer) {
        let interval_seconds = self
            .min_interval_seconds
            .max(final_horizon_time / self.max_count as f64);
        self.state.lock().unwrap().interval_seconds = interval_seconds;
        log!(
            "[{:.4}] [candidate-emit] interval={:.3}s, final_horizon_time={:.3}s, max_count={}",
            _timer.elapsed_seconds(),
            interval_seconds,
            final_horizon_time,
            self.max_count,
        );
    }

    pub(super) fn emit(&self, state: &OptimizeState, timer: Timer, force: bool) {
        let elapsed = timer.elapsed_seconds();
        let mut emit_state = self.state.lock().unwrap();
        if !force && elapsed - emit_state.last_emitted < emit_state.interval_seconds {
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
            Err(_err) => {
                log!(
                    "[{:.4}] [candidate-emit] score={:.3}, force={}, status=serialize-failed, error={}, total={:.4}s",
                    timer.elapsed_seconds(),
                    state.objective,
                    force,
                    _err,
                    timer.elapsed_seconds() - emit_started,
                );
                return;
            }
        };
        let _serialize_seconds = timer.elapsed_seconds() - serialize_started;

        let write_started = timer.elapsed_seconds();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();
        let succeeded = writeln!(stdout, "{output}").is_ok() && stdout.flush().is_ok();
        let _write_seconds = timer.elapsed_seconds() - write_started;
        let _total_seconds = timer.elapsed_seconds() - emit_started;
        if succeeded {
            emit_state.last_emitted = elapsed;
        }
        log!(
            "[{:.4}] [candidate-emit] score={:.3}, force={}, status={}, bytes={}, serialize={:.6}s, write={:.6}s, total={:.6}s",
            timer.elapsed_seconds(),
            state.objective,
            force,
            if succeeded { "ok" } else { "write-failed" },
            output.len(),
            _serialize_seconds,
            _write_seconds,
            _total_seconds,
        );
    }
}
