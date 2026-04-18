use opencv::core::MatTraitConst;
use thiserror::Error;

use crate::capture::Frame;

/// Crop an area of the frame.
/// defined by normalized coordinates (0.0 - 1.0).
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Crop {
    pub top_left: (f32, f32),
    pub bottom_right: (f32, f32),
}

impl Crop {
    pub const fn full() -> Self {
        Self {
            top_left: (0., 0.),
            bottom_right: (1., 1.),
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Calibration {
    /// In radians
    pub rotation: f32,
    pub brightness: f64,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub crop: Crop,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            rotation: 0.,
            brightness: 0.66,
            horizontal_flip: false,
            vertical_flip: false,
            crop: Crop::full(),
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum CalibrationError {
    #[error("Internal error: {0}")]
    Internal(String),
}

pub struct FrameCalibrator {
    frame: Frame,
    calibration: Calibration,
}

impl FrameCalibrator {
    pub fn new() -> Self {
        Self {
            frame: Frame::empty(),
            calibration: Calibration::default(),
        }
    }

    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    pub fn set_calibration(&mut self, calibration: Calibration) {
        self.calibration = calibration;
    }

    pub fn calibrate(&mut self, source: &Frame) -> Result<&Frame, CalibrationError> {
        // TODO: Other things

        // TODO: Double check, maybe `1. - brightness`
        source
            .mat
            .convert_to(
                &mut self.frame.mat,
                source.mat.typ(),
                self.calibration.brightness,
                0.,
            )
            .map_err(|e| CalibrationError::Internal(e.to_string()))?;

        Ok(&self.frame)
    }
}
