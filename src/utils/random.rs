use rand_pcg::rand_core::{RngCore, SeedableRng};

pub trait Random {
    fn next_u32(&mut self) -> u32;

    #[inline(always)]
    fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / ((1u64 << 32) as f64)
    }

    #[inline(always)]
    fn gen_index(&mut self, len: usize) -> usize {
        debug_assert!(len as u64 <= 1 << 32);
        ((len as u64 * self.next_u32() as u64) >> 32) as usize
    }

    #[inline(always)]
    fn gen_range(&mut self, l: usize, r: usize) -> usize {
        debug_assert!(l < r);
        debug_assert!(r as u64 <= 1 << 32);
        l + (((r - l) as u64 * self.next_u32() as u64) >> 32) as usize
    }

    #[inline(always)]
    fn gen_range_lower(&mut self, l: usize, r: usize, shape: f64) -> usize {
        debug_assert!(l < r);
        let span = r - l;
        let u = self.next_f64().powf(shape);
        l + ((u * span as f64) as usize).min(span - 1)
    }

    #[inline(always)]
    fn gen_range_centered(&mut self, l: usize, r: usize, shape: f64) -> usize {
        debug_assert!(l < r);
        let span = r - l;
        let u = self.next_f64() * 2.0 - 1.0;
        let centered = u.signum() * u.abs().powf(shape);
        let normalized = (centered + 1.0) * 0.5;
        l + ((normalized * span as f64) as usize).min(span - 1)
    }

    #[inline(always)]
    fn gen_range_f64(&mut self, l: f64, r: f64) -> f64 {
        debug_assert!(l <= r);
        l + self.next_u32() as f64 * ((r - l) / ((1u64 << 32) as f64))
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

pub fn sample_weighted_index(rng: &mut impl Random, weights: &[f64]) -> usize {
    let total = weights.iter().sum::<f64>();
    let mut value = rng.next_f64() * total;
    for (index, &weight) in weights.iter().enumerate() {
        if value < weight {
            return index;
        }
        value -= weight;
    }
    0
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
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }
}
