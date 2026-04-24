use rand::Rng;

use crate::train::FloatImage;

#[derive(Debug, Clone, Copy)]
pub struct IntensityParams {
    /// Magnitude of the brightness offset.
    pub brightness_range: f32,
    /// Magnitude of the contrast jitter.
    pub contrast_range: f32,
}

impl Default for IntensityParams {
    fn default() -> Self {
        Self {
            brightness_range: 0.2,
            contrast_range: 0.2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IntensityPlan {
    pub brightness: f32,
    pub contrast: f32,
}

impl IntensityPlan {
    pub fn sample<R: Rng + ?Sized>(rng: &mut R, params: &IntensityParams) -> Self {
        let brightness = rng.gen_range(-params.brightness_range..params.brightness_range);
        let contrast = 1.0 + rng.gen_range(-params.contrast_range..params.contrast_range);
        Self {
            brightness,
            contrast,
        }
    }

    pub fn apply_in_place(&self, img: &mut FloatImage) {
        for v in img.as_mut() {
            *v = *v * self.contrast + self.brightness;
        }
    }
}
