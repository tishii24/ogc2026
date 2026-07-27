use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    #[inline]
    pub fn elapsed_seconds(self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}
