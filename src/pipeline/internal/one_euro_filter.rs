use std::{f32::consts::PI, time::Instant};

use crate::pipeline::FilterParameters;

pub struct OneEuroFilter {
    pub parameters: FilterParameters,
    t_prev: Option<Instant>,
    x_prev: Vec<f32>,
    dx_prev: Vec<f32>,
}

impl OneEuroFilter {
    pub fn new(count: usize) -> Self {
        Self {
            parameters: FilterParameters {
                enable: true,
                min_cutoff: 0.5,
                beta: 3.0,
            },
            t_prev: None,
            x_prev: vec![0.; count],
            dx_prev: vec![0.; count],
        }
    }

    pub fn filter(&mut self, x: &[f32]) -> &[f32] {
        assert_eq!(self.x_prev.len(), x.len());

        let now = Instant::now();

        let dt = if let Some(t_prev) = self.t_prev {
            now.duration_since(t_prev).as_secs_f32()
        } else {
            self.x_prev.copy_from_slice(x);
            // We don't need to fill dx_prev as it starts already filled with zeroes.
            self.t_prev = Some(now);
            return &self.x_prev;
        };

        self.t_prev = Some(now);

        if dt <= 0. || !self.parameters.enable {
            self.x_prev.copy_from_slice(x);
            return &self.x_prev;
        }

        let d_cutoff = 1.;
        let a_d = smoothing_factor(dt, d_cutoff);

        for i in 0..x.len() {
            let dx = (x[i] - self.x_prev[i]) / dt;
            let dx_hat = exponential_smoothing(a_d, dx, self.dx_prev[i]);
            let cutoff = self.parameters.min_cutoff + self.parameters.beta * dx_hat.abs();
            let a = smoothing_factor(dt, cutoff);
            let x_hat = exponential_smoothing(a, x[i], self.x_prev[i]);

            self.x_prev[i] = x_hat;
            self.dx_prev[i] = dx_hat;
        }

        &self.x_prev
    }
}

#[inline]
fn smoothing_factor(dt: f32, cutoff: f32) -> f32 {
    let r = 2.0 * PI * cutoff * dt;
    r / (r + 1.0)
}

#[inline]
fn exponential_smoothing(a: f32, x: f32, prev: f32) -> f32 {
    a * x + (1. - a) * prev
}
