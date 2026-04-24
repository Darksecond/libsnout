use thiserror::Error;

use crate::capture::Frame;

/// Crop an area of the frame.
/// defined by normalized coordinates (0.0 - 1.0).
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Crop {
    pub top: f32,
    pub left: f32,
    pub bottom: f32,
    pub right: f32,
}

impl Crop {
    pub const fn full() -> Self {
        Self {
            top: 0.,
            left: 0.,
            bottom: 1.,
            right: 1.,
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PreprocessConfig {
    /// In radians
    pub rotation: f32,
    pub brightness: f32,
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
            frame: Frame::empty(0, 0),
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
        if self.frame.width() != source.width() || self.frame.height() != source.height() {
            self.frame = Frame::empty(source.width() as u32, source.height() as u32);
        }

        // TODO: Other things

        // Brightness
        // TODO: Double check, maybe `1. - brightness`
        {
            let brightness = self.config.brightness;
            let src = source.as_slice();
            let dst: &mut [u8] = self.frame.image.as_mut();
            for (dst, &src) in dst.iter_mut().zip(src.iter()) {
                *dst = (src as f32 * brightness).clamp(0.0, 255.0) as u8
            }
        }

        Ok(&self.frame)
    }
}
