pub mod time {
    use std::time::Instant;

    static mut START: Option<Instant> = None;
    static mut R: f64 = 1.0;

    #[allow(unused)]
    pub fn start_clock(r: f64) {
        unsafe {
            R = r;
            START = Some(Instant::now());
        }
    }

    #[inline]
    #[allow(unused)]
    pub fn elapsed_seconds() -> f64 {
        unsafe {
            match START {
                Some(start) => start.elapsed().as_secs_f64() * R,
                None => {
                    START = Some(Instant::now());
                    0.0
                }
            }
        }
    }
}
