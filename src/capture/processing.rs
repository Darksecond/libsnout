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
pub struct PreprocessConfig {
    /// In radians
    pub rotation: f32,
    pub brightness: f64,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub crop: Crop,
}

impl Default for PreprocessConfig {
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
pub enum PreprocessError {
    #[error("Internal error: {0}")]
    Internal(String),
}

pub struct FramePreprocessor {
    frame: Frame,
    config: PreprocessConfig,
}

impl FramePreprocessor {
    pub fn new() -> Self {
        Self {
            frame: Frame::empty(),
            config: PreprocessConfig::default(),
        }
    }

    pub fn config(&self) -> &PreprocessConfig {
        &self.config
    }

    pub fn set_config(&mut self, calibration: PreprocessConfig) {
        self.config = calibration;
    }

    pub fn process(&mut self, source: &Frame) -> Result<&Frame, PreprocessError> {
        // TODO: Other things

        // TODO: Double check, maybe `1. - brightness`
        source
            .mat
            .convert_to(
                &mut self.frame.mat,
                source.mat.typ(),
                self.config.brightness,
                0.,
            )
            .map_err(|e| PreprocessError::Internal(e.to_string()))?;

        Ok(&self.frame)
    }
}
