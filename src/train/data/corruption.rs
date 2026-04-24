use std::collections::VecDeque;

use image::GrayImage;

pub const DEFAULT_THRESHOLD: f32 = 0.022669;
pub const DEFAULT_ADAPTATION_WINDOW: usize = 100;

const ADAPTIVE_MIN_SAMPLES: usize = 20;
const ADAPTIVE_LOW_FACTOR: f32 = 0.5;
const ADAPTIVE_HIGH_FACTOR: f32 = 3.0;

/// Outcome of a single-frame corruption check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    pub is_corrupted: bool,
    /// Row-pattern-consistency metric for this frame.
    pub value: f32,
    /// Threshold the metric was compared against (possibly adaptive).
    pub threshold: f32,
}

/// Outcome of checking a left/right frame pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairVerdict {
    pub left: Verdict,
    pub right: Verdict,
}

impl PairVerdict {
    /// Either side marked corrupted.
    pub fn any_corrupted(&self) -> bool {
        self.left.is_corrupted || self.right.is_corrupted
    }
}

pub fn calculate_row_pattern_consistency(image: &GrayImage) -> f32 {
    let w = image.width() as usize;
    let h = image.height() as usize;
    if h < 2 || w == 0 {
        return 0.0;
    }

    let raw = image.as_raw();

    // Row means in the normalized [0, 1] range.
    let mut row_means = Vec::with_capacity(h);
    let norm = 1.0f32 / (w as f32 * 255.0f32);
    for row in 0..h {
        let start = row * w;
        let end = start + w;
        let sum: u32 = raw[start..end].iter().map(|&p| p as u32).sum();
        row_means.push(sum as f32 * norm);
    }

    let diffs: Vec<f32> = row_means.windows(2).map(|w| w[1] - w[0]).collect();

    if diffs.is_empty() {
        return 0.0;
    }

    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let var = diffs
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / diffs.len() as f32;
    var.sqrt()
}

#[derive(Debug, Clone)]
pub struct CorruptionDetector {
    base_threshold: f32,
    current_threshold: f32,
    use_adaptive: bool,
    adaptation_window: usize,
    recent_values: VecDeque<f32>,
}

impl Default for CorruptionDetector {
    fn default() -> Self {
        Self::new(DEFAULT_THRESHOLD, true, DEFAULT_ADAPTATION_WINDOW)
    }
}

impl CorruptionDetector {
    pub fn new(base_threshold: f32, use_adaptive: bool, adaptation_window: usize) -> Self {
        let window = adaptation_window.max(1);
        Self {
            base_threshold,
            current_threshold: base_threshold,
            use_adaptive,
            adaptation_window: window,
            recent_values: VecDeque::with_capacity(window),
        }
    }

    pub fn is_corrupted(&mut self, image: &GrayImage) -> Verdict {
        let value = calculate_row_pattern_consistency(image);
        self.update_adaptive_threshold(value);
        Verdict {
            is_corrupted: value > self.current_threshold,
            value,
            threshold: self.current_threshold,
        }
    }

    pub fn process_frame_pair(&mut self, left: &GrayImage, right: &GrayImage) -> PairVerdict {
        let left_v = self.is_corrupted(left);
        let right_v = self.is_corrupted(right);

        PairVerdict {
            left: left_v,
            right: right_v,
        }
    }

    fn update_adaptive_threshold(&mut self, value: f32) {
        if !self.use_adaptive {
            return;
        }

        if self.recent_values.len() == self.adaptation_window {
            self.recent_values.pop_front();
        }
        self.recent_values.push_back(value);

        if self.recent_values.len() < ADAPTIVE_MIN_SAMPLES {
            return;
        }

        let median = median(&self.recent_values);

        let mad_values: Vec<f32> = self
            .recent_values
            .iter()
            .map(|v| (v - median).abs())
            .collect();
        let mad = median_of_slice(&mad_values);

        let adaptive = median + 3.0 * mad;
        let lo = self.base_threshold * ADAPTIVE_LOW_FACTOR;
        let hi = self.base_threshold * ADAPTIVE_HIGH_FACTOR;
        self.current_threshold = adaptive.clamp(lo, hi);
    }
}

fn median(values: &VecDeque<f32>) -> f32 {
    let mut scratch: Vec<f32> = values.iter().copied().collect();
    median_of_slice_mut(&mut scratch)
}

fn median_of_slice(values: &[f32]) -> f32 {
    let mut scratch = values.to_vec();
    median_of_slice_mut(&mut scratch)
}

fn median_of_slice_mut(values: &mut [f32]) -> f32 {
    debug_assert!(!values.is_empty(), "median of empty slice");
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}
