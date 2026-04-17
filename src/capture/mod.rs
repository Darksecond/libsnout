pub mod discovery;
pub mod mono;
pub mod stereo;

mod internal;

use opencv::core::{CV_8UC1, Mat, MatTraitConst, MatTraitConstManual};
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct Frame {
    pub(crate) mat: Mat,
}

impl Frame {
    pub fn new(mat: Mat) -> Self {
        assert_eq!(mat.channels(), 1);
        assert_eq!(mat.typ(), CV_8UC1);

        // TODO: Optional we can *make* it continuous here
        assert!(mat.is_continuous());

        Self { mat }
    }

    pub unsafe fn new_unchecked(mat: Mat) -> Self {
        Self { mat }
    }

    pub fn width(&self) -> usize {
        self.mat.cols() as usize
    }

    pub fn height(&self) -> usize {
        self.mat.rows() as usize
    }

    pub fn as_slice(&self) -> &[u8] {
        self.mat.data_typed::<u8>().expect("Failed to get data")
    }

    pub fn into_mat(self) -> Mat {
        self.mat
    }
}

#[derive(Clone, Debug, Error)]
pub enum CameraError {
    #[error("Failed to open camera device")]
    OpenError,
    #[error("Camera disconnected")]
    Disconnected,
    #[error("Received an empty or invalid frame from hardware")]
    InvalidFrame,
    #[error("Internal driver error: {0}")]
    Internal(String),
}

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
    pub gamma: f64,
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
    pub crop: Crop,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            rotation: 0.,
            gamma: 1.,
            horizontal_flip: false,
            vertical_flip: false,
            crop: Crop::full(),
        }
    }
}
