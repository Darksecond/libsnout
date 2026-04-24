use imageproc::filter::gaussian_blur_f32;
use rand::Rng;

use crate::train::FloatImage;

#[derive(Debug, Clone, Copy)]
pub struct BlurParams {
    /// Minimum sigma (inclusive). Must be > 0.0.
    pub sigma_min: f32,
    /// Maximum sigma (exclusive).
    pub sigma_max: f32,
}

impl Default for BlurParams {
    fn default() -> Self {
        Self {
            sigma_min: 0.1,
            sigma_max: 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlurPlan {
    pub sigma: f32,
}

impl BlurPlan {
    pub fn sample<R: Rng + ?Sized>(rng: &mut R, params: &BlurParams) -> Self {
        let sigma = rng.gen_range(params.sigma_min..params.sigma_max);
        Self { sigma }
    }

    pub fn apply(&self, img: &FloatImage) -> FloatImage {
        if self.sigma > 0.0 {
            gaussian_blur_f32(img, self.sigma)
        } else {
            img.clone()
        }
    }
}
