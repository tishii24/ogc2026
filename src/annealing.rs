use std::sync::Mutex;

use crate::util::{
    rand::{RandPcg64Mcg, Random},
    time::Timer,
};

pub struct SharedBest<T> {
    inner: Mutex<(f64, T)>,
}

impl<T: Clone> SharedBest<T> {
    pub fn new(key: f64, state: T) -> Self {
        Self {
            inner: Mutex::new((key, state)),
        }
    }

    pub fn update(&self, key: f64, state: &T) -> bool {
        let mut best = self.inner.lock().unwrap();
        if key + 1e-9 >= best.0 {
            return false;
        }
        best.0 = key;
        best.1.clone_from(state);
        true
    }

    pub fn get_if_better(&self, current_key: f64) -> Option<(f64, T)> {
        let best = self.inner.lock().unwrap();
        (best.0 + 1e-9 < current_key).then(|| (best.0, best.1.clone()))
    }

    pub fn key(&self) -> f64 {
        self.inner.lock().unwrap().0
    }

    pub fn into_inner(self) -> T {
        self.inner.into_inner().unwrap().1
    }
}

pub struct TemperatureSchedule {
    start_time: f64,
    deadline: f64,
    start_temperature: f64,
    end_temperature: f64,
}

impl TemperatureSchedule {
    pub fn new(
        start_time: f64,
        deadline: f64,
        start_temperature: f64,
        end_temperature: f64,
    ) -> Self {
        Self {
            start_time,
            deadline,
            start_temperature,
            end_temperature,
        }
    }

    pub fn temperature(&self, elapsed: f64) -> f64 {
        let progress = ((elapsed - self.start_time) / (self.deadline - self.start_time).max(1e-4))
            .clamp(0.0, 1.0);
        self.start_temperature * (self.end_temperature / self.start_temperature).powf(progress)
    }
}

pub struct AnnealingWorkerContext {
    pub rng: RandPcg64Mcg,
    temperature: TemperatureSchedule,
    deadline: f64,
    iterations: usize,
}

impl AnnealingWorkerContext {
    pub fn new(
        timer: Timer,
        deadline: f64,
        start_temperature: f64,
        end_temperature: f64,
        rng_seed: u64,
    ) -> Self {
        Self {
            rng: RandPcg64Mcg::new(rng_seed),
            temperature: TemperatureSchedule::new(
                timer.elapsed_seconds(),
                deadline,
                start_temperature,
                end_temperature,
            ),
            deadline,
            iterations: 0,
        }
    }

    pub fn next(&mut self, timer: Timer) -> Option<f64> {
        let elapsed = timer.elapsed_seconds();
        if elapsed >= self.deadline {
            return None;
        }
        self.iterations += 1;
        Some(self.temperature.temperature(elapsed))
    }

    pub fn should_exchange(&self, interval: usize) -> bool {
        self.iterations % interval == 0
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

pub fn accept(delta: f64, temperature: f64, rng: &mut impl Random) -> bool {
    delta <= 0.0 || rng.nextf() < (-delta / temperature).exp()
}
