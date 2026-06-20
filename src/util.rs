#![allow(dead_code)]

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

use rand_pcg::rand_core::{RngCore, SeedableRng};

pub trait Random {
    fn _next(&mut self) -> u32;

    #[inline(always)]
    fn nextf(&mut self) -> f64 {
        self._next() as f64 / ((1u64 << 32) as f64)
    }

    #[inline(always)]
    fn choice<T: Clone + Copy>(&mut self, v: &[T]) -> T {
        let idx = self.gen_index(v.len());
        v[idx]
    }

    #[inline(always)]
    fn gen_index(&mut self, len: usize) -> usize {
        debug_assert!(len as u64 <= 1 << 32);
        ((len as u64 * self._next() as u64) >> 32) as usize
    }

    #[inline(always)]
    fn gen_range(&mut self, l: usize, r: usize) -> usize {
        debug_assert!(l < r);
        debug_assert!(r as u64 <= 1 << 32);
        l + (((r - l) as u64 * self._next() as u64) >> 32) as usize
    }

    #[inline(always)]
    fn gen_rangef(&mut self, l: f64, r: f64) -> f64 {
        debug_assert!(l <= r);
        l + self._next() as f64 * ((r - l) / ((1u64 << 32) as f64))
    }

    #[inline(always)]
    fn shuffle<T>(&mut self, v: &mut [T]) {
        let n = v.len();
        for i in (1..n).rev() {
            let j = self.gen_range(0, i + 1);
            v.swap(i, j);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct XorShift32 {
    state: u32,
}

impl XorShift32 {
    pub fn new(seed: u32) -> Self {
        assert_ne!(seed, 0);
        Self { state: seed }
    }
}

impl Random for XorShift32 {
    #[inline(always)]
    fn _next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }
}

#[derive(Debug, Clone)]
pub struct RandPcg64Mcg {
    inner: rand_pcg::Pcg64Mcg,
}

impl RandPcg64Mcg {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: rand_pcg::Pcg64Mcg::seed_from_u64(seed),
        }
    }
}

impl Random for RandPcg64Mcg {
    #[inline(always)]
    fn _next(&mut self) -> u32 {
        self.inner.next_u32()
    }
}

#[derive(Debug, Clone)]
pub struct BufferedRandom {
    buf: Vec<u32>,
    pos: usize,
}

impl BufferedRandom {
    pub fn new<R: Random>(rnd: &mut R, buf_size: usize) -> Self {
        assert!(0 < buf_size);
        assert!(buf_size < 1_000_000);
        let mut buf = Vec::with_capacity(buf_size);
        for _ in 0..buf_size {
            buf.push(rnd._next());
        }
        Self { buf, pos: 0 }
    }
}

impl Random for BufferedRandom {
    #[inline(always)]
    fn _next(&mut self) -> u32 {
        let v = self.buf[self.pos];
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
        v
    }
}

pub trait RandomSampler<T> {
    fn sample(&mut self) -> T;
}

pub struct DiscreteSampler<T, R> {
    buf: Vec<T>,
    rnd: R,
}

impl<T: Copy, R: Random> DiscreteSampler<T, R> {
    pub fn new(weight_values: &[(usize, T)], rnd: R) -> Self {
        let weight_sum = weight_values.iter().map(|(w, _)| *w).sum::<usize>();
        assert!(0 < weight_sum);
        assert!(weight_sum < 1_000_000);
        let mut buf = Vec::with_capacity(weight_sum);
        for &(w, val) in weight_values.iter() {
            buf.extend(std::iter::repeat_n(val, w));
        }
        Self { buf, rnd }
    }
}

impl<T: Copy, R: Random> RandomSampler<T> for DiscreteSampler<T, R> {
    fn sample(&mut self) -> T {
        self.rnd.choice(&self.buf)
    }
}

pub struct ContinousSampler<R: Random> {
    buf: Vec<f64>,
    rnd: R,
}

impl<R: Random> ContinousSampler<R> {
    pub fn new<F>(f: F, x_min: f64, x_max: f64, size: usize, rnd: R) -> Self
    where
        F: Fn(f64) -> f64,
    {
        assert!(0 < size);
        assert!(size < 1_000_000);
        let mut buf = Vec::with_capacity(size);
        let step = (x_max - x_min) / (size as f64 - 1.);
        for i in 0..size {
            let x = x_min + step * (i as f64);
            buf.push(f(x));
        }
        Self { buf, rnd }
    }
}

impl<R: Random> RandomSampler<f64> for ContinousSampler<R> {
    fn sample(&mut self) -> f64 {
        self.rnd.choice(&self.buf)
    }
}
