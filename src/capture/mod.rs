pub mod discovery;
mod mono;
pub mod processing;
mod stereo;

mod internal;

use image::GrayImage;
use thiserror::Error;

pub use mono::MonoCamera;
pub use stereo::StereoCamera;

#[derive(Debug)]
pub struct Frame {
    pub(crate) image: GrayImage,
}

impl Frame {
    pub fn new(image: GrayImage) -> Self {
        Self { image }
    }

    pub fn empty(width: u32, height: u32) -> Self {
        Self {
            image: GrayImage::new(width, height),
        }
    }

    pub unsafe fn new_unchecked(image: GrayImage) -> Self {
        Self { image }
    }

    pub fn width(&self) -> usize {
        self.image.width() as _
    }

    pub fn height(&self) -> usize {
        self.image.height() as usize
    }

    pub fn as_slice(&self) -> &[u8] {
        self.image.iter().as_slice()
    }

    pub fn into_image(self) -> GrayImage {
        self.image
    }
}

#[derive(Clone, Debug, Error)]
pub enum CameraError {
    #[error("Failed to open camera device")]
    OpenError,
    /// Received an empty or invalid frame from hardware.
    /// This can mean that the camera is disconnected or the frame data is corrupted.
    #[error("Received an empty or invalid frame from hardware")]
    InvalidFrame,
    #[error("Internal driver error: {0}")]
    Internal(String),
    #[error("Frame size mismatch: expected {expected:?}, got {actual:?}")]
    FrameMismatch {
        expected: (u32, u32),
        actual: (u32, u32),
    },
}

impl From<opencv::Error> for CameraError {
    fn from(e: opencv::Error) -> Self {
        Self::Internal(e.to_string())
    }
}
