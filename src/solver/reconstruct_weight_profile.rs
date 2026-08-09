use std::sync::Mutex;

const WEIGHT_COUNT: usize = 6;
const BIN_COUNT: usize = 8;
const MIN_PAIR_ATTEMPTS: u64 = 100;
const WEIGHT_NAMES: [&str; WEIGHT_COUNT] = [
    "volume",
    "pref_spread",
    "limit_time_urgency",
    "slack_tightness",
    "current_penalty",
    "random",
];

#[derive(Clone, Copy, Default)]
struct OutcomeCounts {
    failed: u64,
    unchanged: u64,
    changed: u64,
}

impl OutcomeCounts {
    fn add(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Failed => self.failed += 1,
            Outcome::Unchanged => self.unchanged += 1,
            Outcome::Changed => self.changed += 1,
        }
    }

    fn attempts(self) -> u64 {
        self.failed + self.unchanged + self.changed
    }

    fn changed_rate(self) -> f64 {
        if self.attempts() > 0 {
            100.0 * self.changed as f64 / self.attempts() as f64
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Outcome {
    Failed,
    Unchanged,
    Changed,
}

struct WeightStats {
    total: OutcomeCounts,
    marginal: Vec<OutcomeCounts>,
    pairs: Vec<OutcomeCounts>,
    top2: Vec<OutcomeCounts>,
    top3: Vec<OutcomeCounts>,
}

impl WeightStats {
    fn new() -> Self {
        Self {
            total: OutcomeCounts::default(),
            marginal: vec![OutcomeCounts::default(); WEIGHT_COUNT * BIN_COUNT],
            pairs: vec![
                OutcomeCounts::default();
                WEIGHT_COUNT * WEIGHT_COUNT * BIN_COUNT * BIN_COUNT
            ],
            top2: vec![OutcomeCounts::default(); 1 << WEIGHT_COUNT],
            top3: vec![OutcomeCounts::default(); 1 << WEIGHT_COUNT],
        }
    }

    fn record(&mut self, weights: [f64; WEIGHT_COUNT], outcome: Outcome) {
        self.total.add(outcome);
        let bins = weights.map(|weight| ((weight * BIN_COUNT as f64) as usize).min(BIN_COUNT - 1));
        for (index, &bin) in bins.iter().enumerate() {
            self.marginal[index * BIN_COUNT + bin].add(outcome);
        }
        for a in 0..WEIGHT_COUNT {
            for b in a + 1..WEIGHT_COUNT {
                let index = (((a * WEIGHT_COUNT + b) * BIN_COUNT + bins[a]) * BIN_COUNT) + bins[b];
                self.pairs[index].add(outcome);
            }
        }

        let mut order: Vec<usize> = (0..WEIGHT_COUNT).collect();
        order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]).then(a.cmp(&b)));
        let top2_mask = (1usize << order[0]) | (1usize << order[1]);
        let top3_mask = top2_mask | (1usize << order[2]);
        self.top2[top2_mask].add(outcome);
        self.top3[top3_mask].add(outcome);
    }
}

static STATS: Mutex<Option<WeightStats>> = Mutex::new(None);

pub(super) fn reset() {
    *STATS.lock().unwrap() = Some(WeightStats::new());
}

pub(super) fn record(weights: [f64; WEIGHT_COUNT], outcome: Outcome) {
    if let Some(stats) = STATS.lock().unwrap().as_mut() {
        stats.record(weights, outcome);
    }
}

fn mask_name(mask: usize) -> String {
    WEIGHT_NAMES
        .iter()
        .enumerate()
        .filter(|(index, _)| mask & (1 << index) != 0)
        .map(|(_, name)| *name)
        .collect::<Vec<_>>()
        .join("+")
}

pub(super) fn log_stats(horizon_index: usize) {
    let stats = STATS.lock().unwrap();
    let Some(stats) = stats.as_ref() else {
        return;
    };
    let mut output = format!(
        "[reconstruct-weight] horizon={} attempts={} changed={} unchanged={} failed={} changed_rate={:.2}%",
        horizon_index,
        stats.total.attempts(),
        stats.total.changed,
        stats.total.unchanged,
        stats.total.failed,
        stats.total.changed_rate(),
    );

    for (weight_index, name) in WEIGHT_NAMES.iter().enumerate() {
        output.push_str(&format!("\n  marginal {name}:"));
        for bin in 0..BIN_COUNT {
            let counts = stats.marginal[weight_index * BIN_COUNT + bin];
            output.push_str(&format!(
                " {}:[{},{},{:.2}%]",
                bin,
                counts.attempts(),
                counts.changed,
                counts.changed_rate(),
            ));
        }
    }

    output.push_str("\n  top2:");
    for (mask, &counts) in stats.top2.iter().enumerate() {
        if counts.attempts() > 0 {
            output.push_str(&format!(
                " {}:[{},{},{:.2}%]",
                mask_name(mask),
                counts.attempts(),
                counts.changed,
                counts.changed_rate(),
            ));
        }
    }
    output.push_str("\n  top3:");
    for (mask, &counts) in stats.top3.iter().enumerate() {
        if counts.attempts() > 0 {
            output.push_str(&format!(
                " {}:[{},{},{:.2}%]",
                mask_name(mask),
                counts.attempts(),
                counts.changed,
                counts.changed_rate(),
            ));
        }
    }

    let mut pair_cells = Vec::new();
    for a in 0..WEIGHT_COUNT {
        for b in a + 1..WEIGHT_COUNT {
            for bin_a in 0..BIN_COUNT {
                for bin_b in 0..BIN_COUNT {
                    let index = (((a * WEIGHT_COUNT + b) * BIN_COUNT + bin_a) * BIN_COUNT) + bin_b;
                    let counts = stats.pairs[index];
                    if counts.attempts() >= MIN_PAIR_ATTEMPTS {
                        pair_cells.push((
                            counts.changed_rate(),
                            counts.attempts(),
                            a,
                            b,
                            bin_a,
                            bin_b,
                            counts,
                        ));
                    }
                }
            }
        }
    }
    pair_cells.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
    output.push_str("\n  pair_top20:");
    for &(_, _, a, b, bin_a, bin_b, counts) in pair_cells.iter().take(20) {
        output.push_str(&format!(
            " {}[{}]+{}[{}]:[{},{},{:.2}%]",
            WEIGHT_NAMES[a],
            bin_a,
            WEIGHT_NAMES[b],
            bin_b,
            counts.attempts(),
            counts.changed,
            counts.changed_rate(),
        ));
    }
    log!("{output}");
}
