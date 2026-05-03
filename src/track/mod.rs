use std::path::Path;

use thiserror::Error;

use crate::{
    capture::{CameraError, processing::PreprocessError},
    pipeline::{PipelineError, initialize_runtime_with_path},
};

pub mod eye;
pub mod face;
pub mod output;

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

/// Initialize the ONNX runtime.
///
/// This will try and find the best matching candidate.
///
/// If a path is provided, it will be tried first.
/// `LD_LIBRARY_PATH` is parsed, or a fallback path to `/usr/lib64/libonnxruntime.so` will be used.
///
/// This function will panic if no path could be found.
pub fn initialize_runtime(path: Option<impl AsRef<Path>>) {
    if let Some(path) = path {
        let path = path.as_ref();

        if path.exists() {
            initialize_runtime_with_path(path);
            return;
        } else {
            tracing::error!(path = %path.display(), "path does not exist");
        }
    }

    if let Some(search_path) = std::env::var("LD_LIBRARY_PATH").ok() {
        for dir in search_path.split(':') {
            let path = Path::new(dir).join("libonnxruntime.so");
            if path.exists() {
                initialize_runtime_with_path(&path);
                return;
            }
        }
    }

    let fallback = Path::new("/usr/lib64/libonnxruntime.so");
    if fallback.exists() {
        initialize_runtime_with_path(fallback);
        return;
    }

    panic!("Could not find libonnxruntime.so");
}
