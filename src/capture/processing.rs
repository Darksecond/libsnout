use image::Luma;
use image::imageops::crop_imm;
use imageproc::geometric_transformations::{Interpolation, Projection, warp};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capture::Frame;

/// Pixel-space crop region within a frame.
#[derive(Copy, Clone, Debug)]
struct PixelCrop {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// Specifies a square crop region.
///
/// `major_shift` shifts the cropped region along the longest axis.
/// -1 and +1 correspond to the crop touching opposite edges.
///
/// `minor_shift` shifts it along the shortest axis.
/// This will only have an effect when `scale` is larger than 1.0.
///
/// Both values are in the range [-1.0, 1.0], with 0.0 being centered.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
#[serde(default)]
pub struct Crop {
    pub major_shift: f32,
    pub minor_shift: f32,
    pub scale: f32,
}

impl Default for Crop {
    fn default() -> Self {
        Self {
            major_shift: 0.0,
            minor_shift: 0.0,
            scale: 1.0,
        }
    }
}

impl Crop {
    fn to_pixel_crop(self, src_w: u32, src_h: u32) -> PixelCrop {
        let minor = src_w.min(src_h) as f32;
        let side = (minor / self.scale).clamp(1.0, minor).round() as u32;

        let (major_axis_len, minor_axis_len, landscape) = if src_w >= src_h {
            (src_w, src_h, true)
        } else {
            (src_h, src_w, false)
        };

        let major_slack = major_axis_len.saturating_sub(side) as f32;
        let minor_slack = minor_axis_len.saturating_sub(side) as f32;

        // shift in [-1, 1] → offset in [0, slack]
        let major_off =
            ((1.0 + self.major_shift.clamp(-1.0, 1.0)) * 0.5 * major_slack).round() as u32;
        let minor_off =
            ((1.0 + self.minor_shift.clamp(-1.0, 1.0)) * 0.5 * minor_slack).round() as u32;

        let (x, y) = if landscape {
            (major_off, minor_off)
        } else {
            (minor_off, major_off)
        };

        PixelCrop {
            x,
            y,
            width: side,
            height: side,
        }
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[repr(C)]
#[serde(default)]
pub struct PreprocessConfig {
    /// In degrees
    pub rotation: f32,
    pub brightness: f32,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            rotation: 0.,
            brightness: 0.66,
            horizontal_flip: false,
            vertical_flip: false,
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
    crop: Crop,
}

impl FramePreprocessor {
    pub fn new() -> Self {
        Self {
            frame: Frame::empty(0, 0),
            config: PreprocessConfig::default(),
            crop: Crop::default(),
        }
    }

    pub fn config(&self) -> &PreprocessConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PreprocessConfig) {
        self.config = config;
    }

    pub fn crop(&self) -> Crop {
        self.crop
    }

    pub fn set_crop(&mut self, crop: Crop) {
        self.crop = crop;
    }

    pub fn process(&mut self, source: &Frame) -> Result<&Frame, PreprocessError> {
        // Crop
        let pc = self
            .crop
            .to_pixel_crop(source.width() as u32, source.height() as u32);
        self.frame.image = crop_imm(&source.image, pc.x, pc.y, pc.width, pc.height).to_image();

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
            let rotation = self.config.rotation.to_radians();
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
