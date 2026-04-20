mod eye;
mod face;

pub use eye::{EyeCalibrator, EyeShape};
pub use face::{FaceShape, ManualFaceCalibrator};

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Bounds {
    pub min: f32,
    pub max: f32,
    pub lower: f32,
    pub upper: f32,
}

impl Bounds {
    pub(crate) const fn new_01() -> Self {
        Self {
            min: 0.,
            max: 1.,
            lower: 0.,
            upper: 1.,
        }
    }

    pub(crate) const fn new_11() -> Self {
        Self {
            min: -1.,
            max: 1.,
            lower: -1.,
            upper: 1.,
        }
    }

    pub(crate) const fn remap(&self, value: f32) -> f32 {
        self.min + (value - self.lower) * (self.max - self.min) / (self.upper - self.lower)
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct ShapeWeight<S> {
    pub shape: S,

    /// The value with the filter applied.
    /// This is probably what you want to use.
    pub value: f32,
}
