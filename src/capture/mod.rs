pub mod discovery;
mod mono;
mod stereo;

mod internal;

use opencv::core::{CV_8UC1, Mat, MatTraitConst, MatTraitConstManual};
use thiserror::Error;

pub use mono::MonoCamera;
pub use stereo::StereoCamera;

#[derive(Debug)]
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

    /// Create an empty Frame
    pub fn empty() -> Self {
        Self {
            mat: Mat::default(),
        }
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
    /// Received an empty or invalid frame from hardware.
    /// This can mean that the camera is disconnected or the frame data is corrupted.
    #[error("Received an empty or invalid frame from hardware")]
    InvalidFrame,
    #[error("Internal driver error: {0}")]
    Internal(String),
}
