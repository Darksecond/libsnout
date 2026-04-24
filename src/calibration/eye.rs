use crate::calibration::{Bounds, ShapeWeight};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EyeShape {
    LeftEyePitch,
    LeftEyeYaw,
    LeftEyeLid,
    RightEyePitch,
    RightEyeYaw,
    RightEyeLid,
}

impl EyeShape {
    pub(crate) const fn from(value: usize) -> Self {
        assert!(value < Self::count());

        unsafe { std::mem::transmute(value as u8) }
    }

    pub(crate) const fn count() -> usize {
        const {
            assert!(Self::RightEyeLid as usize + 1 == 6);
        }

        Self::RightEyeLid as usize + 1
    }

    pub(crate) fn to_etvr(self) -> &'static str {
        match self {
            Self::LeftEyePitch => "/avatar/parameters/v2/EyeLeftX",
            Self::LeftEyeYaw => "/avatar/parameters/v2/EyeLeftY",
            Self::LeftEyeLid => "/avatar/parameters/v2/EyeLidLeft",
            Self::RightEyePitch => "/avatar/parameters/v2/EyeRightX",
            Self::RightEyeYaw => "/avatar/parameters/v2/EyeRightY",
            Self::RightEyeLid => "/avatar/parameters/v2/EyeLidRight",
        }
    }

    pub(crate) fn to_etvr_value(self, value: f32) -> f32 {
        if self == Self::LeftEyeLid || self == Self::RightEyeLid {
            1. - value
        } else {
            value
        }
    }
}

/// Calibrator for eye shapes.
///
/// This will calibrate the raw values out of the pipeline.
/// Only the LeftEyeLid and RightEyeLid bounds are respected.
pub struct EyeCalibrator {
    bounds: Vec<Bounds>,
    weights: Vec<ShapeWeight<EyeShape>>,
    link_eyes: bool,
}

impl EyeCalibrator {
    pub fn new() -> Self {
        let mut bounds = vec![Bounds::new_11(); EyeShape::count()];
        bounds[EyeShape::LeftEyeLid as usize] = Bounds::new_01();
        bounds[EyeShape::RightEyeLid as usize] = Bounds::new_01();

        Self {
            bounds,
            weights: default_weights(),
            link_eyes: true,
        }
    }

    pub fn link_eyes(&self) -> bool {
        self.link_eyes
    }

    pub fn set_link_eyes(&mut self, link_eyes: bool) {
        self.link_eyes = link_eyes;
    }

    pub fn bounds(&self, shape: EyeShape) -> Bounds {
        self.bounds[shape as usize]
    }

    pub fn set_bounds(&mut self, shape: EyeShape, bounds: Bounds) {
        self.bounds[shape as usize] = bounds;
    }

    pub fn calibrate(&mut self, weights: &[f32]) -> &[ShapeWeight<EyeShape>] {
        let mut remapped = [0.; EyeShape::count()];
        self.remap(weights, &mut remapped);

        for (weight, value) in self.weights.iter_mut().zip(remapped) {
            weight.value = value;
        }

        &self.weights
    }

    fn remap(&self, source: &[f32], target: &mut [f32]) {
        let mul_v = 2.;
        let mul_y = 2.;

        let left_pitch = source[0] * mul_y - mul_y / 2.;
        let left_yaw = source[1] * mul_v - mul_v / 2.;
        let left_lid = 1. - source[2];

        let right_pitch = source[3] * mul_y - mul_y / 2.;
        let right_yaw = source[4] * mul_v - mul_v / 2.;
        let right_lid = 1. - source[5];

        let eye_y = (left_pitch * left_lid + right_pitch * right_lid) / (left_lid + right_lid);

        let mut left_eye_yaw_corrected = right_yaw * (1. - left_lid) + left_yaw * left_lid;
        let mut right_eye_yaw_corrected = left_yaw * (1. - right_lid) + right_yaw * right_lid;

        if self.link_eyes {
            let raw_convergence = (right_eye_yaw_corrected - left_eye_yaw_corrected) / 2.;
            let convergence = raw_convergence.max(0.);

            let average_yaw = (right_eye_yaw_corrected + left_eye_yaw_corrected) / 2.;

            left_eye_yaw_corrected = average_yaw - convergence;
            right_eye_yaw_corrected = average_yaw + convergence;
        }

        target[0] = right_eye_yaw_corrected;
        target[1] = eye_y;
        target[2] = self.bounds[2].remap(right_lid);

        target[3] = left_eye_yaw_corrected;
        target[4] = eye_y;
        target[5] = self.bounds[5].remap(left_lid);
    }
}

fn default_weights() -> Vec<ShapeWeight<EyeShape>> {
    let mut weights = Vec::with_capacity(EyeShape::count());

    for index in 0..EyeShape::count() {
        weights.push(ShapeWeight {
            shape: EyeShape::from(index),
            value: 0.0,
        })
    }

    weights
}
