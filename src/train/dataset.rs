mod blur;
mod intensity;
mod spatial;

use std::sync::Arc;

use burn::data::dataset::Dataset;
use rand::Rng;

use crate::models::dual_eye_net::{INPUT_CHANNELS, LABEL_DIMS};
use crate::train::FloatImage;
use crate::train::data::DataReader;
use blur::{BlurParams, BlurPlan};
use intensity::{IntensityParams, IntensityPlan};
use spatial::{SpatialParams, SpatialPlan};

#[derive(Debug, Clone, Copy)]
pub(crate) struct AugmentConfig {
    pub spatial_probability: f32,
    pub spatial: SpatialParams,
    pub intensity_probability: f32,
    pub intensity: IntensityParams,
    pub blur_probability: f32,
    pub blur: BlurParams,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            spatial_probability: 0.2,
            spatial: SpatialParams {
                max_shift: 24,
                max_rotation_deg: 10.0,
                max_scale: 0.1,
            },
            intensity_probability: 0.3,
            intensity: IntensityParams {
                brightness_range: 0.1,
                contrast_range: 0.6,
            },
            blur_probability: 0.1,
            blur: BlurParams {
                sigma_min: 0.5,
                sigma_max: 2.0,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CaptureItem {
    pub image: Vec<f32>,
    pub label: [f32; LABEL_DIMS],
}

#[derive(Clone)]
pub(crate) struct CaptureDataset {
    data: Arc<DataReader>,
    augmented: bool,
    augment: AugmentConfig,
}

impl CaptureDataset {
    pub fn augmented(data: Arc<DataReader>) -> Self {
        Self {
            data,
            augmented: true,
            augment: AugmentConfig::default(),
        }
    }

    pub fn plain(data: Arc<DataReader>) -> Self {
        Self {
            data,
            augmented: false,
            augment: AugmentConfig::default(),
        }
    }

    fn build_item<R: Rng + ?Sized>(&self, idx: usize, rng: &mut R) -> Option<CaptureItem> {
        let frames = self.data.history_window(idx as u32)?;

        let eyes = [
            &frames[0].left,
            &frames[0].right,
            &frames[1].left,
            &frames[1].right,
            &frames[2].left,
            &frames[2].right,
            &frames[3].left,
            &frames[3].right,
        ];

        let label = frames[0].label;

        let image = if !self.augmented {
            self.build_plain_image(eyes)
        } else {
            self.build_augmented_image(eyes, rng)
        };

        Some(CaptureItem { image, label })
    }

    fn build_augmented_image<R: Rng + ?Sized>(
        &self,
        eyes: [&FloatImage; INPUT_CHANNELS],
        rng: &mut R,
    ) -> Vec<f32> {
        let mut channels: Vec<FloatImage> = eyes.iter().map(|&eye| eye.clone()).collect();

        if rng.gen_bool(self.augment.spatial_probability.clamp(0.0, 1.0) as f64) {
            let plan = SpatialPlan::sample(rng, &self.augment.spatial);
            for ch in channels.iter_mut() {
                *ch = plan.apply(ch);
            }
        }

        if rng.gen_bool(self.augment.intensity_probability.clamp(0.0, 1.0) as f64) {
            let plan = IntensityPlan::sample(rng, &self.augment.intensity);
            for ch in channels.iter_mut() {
                plan.apply_in_place(ch);
            }

            normalize_by_max(&mut channels);
        }

        if rng.gen_bool(self.augment.blur_probability.clamp(0.0, 1.0) as f64) {
            let plan = BlurPlan::sample(rng, &self.augment.blur);
            for ch in channels.iter_mut() {
                *ch = plan.apply(ch);
            }
        }

        // Flatten to CHW.
        let per_channel = channels[0].as_raw().len();
        let mut image = Vec::with_capacity(INPUT_CHANNELS * per_channel);
        for ch in &channels {
            image.extend_from_slice(ch.as_raw());
        }

        image
    }

    fn build_plain_image(&self, eyes: [&FloatImage; INPUT_CHANNELS]) -> Vec<f32> {
        let per_channel = eyes[0].as_raw().len();
        let mut image = Vec::with_capacity(INPUT_CHANNELS * per_channel);

        for eye in &eyes {
            image.extend_from_slice(eye.as_raw());
        }

        image
    }
}

impl Dataset<CaptureItem> for CaptureDataset {
    fn get(&self, index: usize) -> Option<CaptureItem> {
        let mut rng = rand::thread_rng();
        self.build_item(index, &mut rng)
    }

    fn len(&self) -> usize {
        self.data.usable_len() as usize
    }
}

/// Divide every pixel in every channel by the largest value observed across *all* channels.
fn normalize_by_max(channels: &mut [FloatImage]) {
    let mut max = 0.0f32;

    for ch in channels.iter() {
        for &v in ch.as_raw() {
            if !v.is_finite() {
                return;
            }
            if v > max {
                max = v;
            }
        }
    }

    if max <= 0.0 {
        return;
    }

    let inv = 1.0 / max;

    for ch in channels.iter_mut() {
        for v in ch.as_mut() {
            *v *= inv;
        }
    }
}
