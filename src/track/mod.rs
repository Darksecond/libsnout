use thiserror::Error;

use crate::{
    capture::{CameraError, processing::PreprocessError},
    pipeline::PipelineError,
};

pub mod face;

#[derive(Clone, Debug, Error)]
pub enum TrackerError {
    #[error("failed to load model: {0}")]
    Model(String),
    #[error("failed to open camera: {0}")]
    Open(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<CameraError> for TrackerError {
    fn from(error: CameraError) -> Self {
        match error {
            CameraError::InvalidFormat(e) => TrackerError::Internal(e),
            CameraError::InvalidFrame(e) => TrackerError::Internal(e),
            CameraError::Internal(e) => TrackerError::Internal(e),
        }
    }
}

impl From<PreprocessError> for TrackerError {
    fn from(error: PreprocessError) -> Self {
        TrackerError::Internal(error.to_string())
    }
}

impl From<PipelineError> for TrackerError {
    fn from(error: PipelineError) -> Self {
        match error {
            PipelineError::Load(e) => TrackerError::Model(e),
            PipelineError::Inference(e) => TrackerError::Internal(e),
        }
    }
}
