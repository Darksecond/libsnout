use image::Luma;
use image::imageops::crop_imm;
use imageproc::geometric_transformations::{Interpolation, Projection, warp};
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

    /// Convert normalised crop coordinates to pixel-space `(x, y, w, h)`,
    /// clamping to source bounds and falling back to the full frame when
    /// the region is empty or already covers the entire source.
    fn to_pixels(&self, (src_w, src_h): (u32, u32)) -> (u32, u32, u32, u32) {
        let x = (self.left.clamp(0.0, 1.0) * src_w as f32) as u32;
        let y = (self.top.clamp(0.0, 1.0) * src_h as f32) as u32;
        let w = ((self.right.clamp(0.0, 1.0) * src_w as f32) as u32).saturating_sub(x);
        let h = ((self.bottom.clamp(0.0, 1.0) * src_h as f32) as u32).saturating_sub(y);

        if w == 0 || h == 0 || (w == src_w && h == src_h) {
            (0, 0, src_w, src_h)
        } else {
            (x, y, w, h)
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
        // Crop
        let (rx, ry, rw, rh) = self.config.crop.to_pixels(source.image.dimensions());
        self.frame.image = crop_imm(&source.image, rx, ry, rw, rh).to_image();

        // Brightness
        // TODO: Double check, maybe `1. - brightness`
        {
            let brightness = self.config.brightness;
            let pixels: &mut [u8] = self.frame.image.as_mut();
            for p in pixels.iter_mut() {
                *p = (*p as f32 * brightness).clamp(0.0, 255.0) as u8;
            }
        }

        // Rotation and flip
        {
            let rotation = self.config.rotation;
            let h_flip = self.config.horizontal_flip;
            let v_flip = self.config.vertical_flip;

            if rotation != 0.0 || h_flip || v_flip {
                let (sin, cos) = (rotation as f64).sin_cos();
                let scale = 1.0 / (cos.abs() + sin.abs());
                let hscale = (if h_flip { -scale } else { scale }) as f32;
                let vscale = (if v_flip { -scale } else { scale }) as f32;

                let w = self.frame.width() as f32;
                let h = self.frame.height() as f32;
                let cx = w / 2.0;
                let cy = h / 2.0;

                let projection = Projection::translate(-cx, -cy)
                    .and_then(Projection::scale(1.0 / hscale, 1.0 / vscale))
                    .and_then(Projection::rotate(rotation))
                    .and_then(Projection::translate(cx, cy));

                self.frame.image = warp(
                    &self.frame.image,
                    &projection,
                    Interpolation::Bilinear,
                    Luma([0u8]),
                );
            }
        }

        Ok(&self.frame)
    }
}
