mod eye;
mod face;

mod internal;

use std::path::Path;

use thiserror::Error;

pub use eye::EyePipeline;
pub use face::FacePipeline;

#[derive(Clone, Debug, Error)]
pub enum PipelineError {
    #[error("Failed to load model: {0}")]
    Load(String),
    #[error("Inference failed: {0}")]
    Inference(String),
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
#[repr(C)]
pub struct FilterParameters {
    pub enable: bool,
    pub min_cutoff: f32,
    pub beta: f32,
}

/// Initialize the ONNX runtime with a specific path.
pub(crate) fn initialize_runtime_with_path(path: impl AsRef<Path>) {
    tracing::info!(path = %path.as_ref().display(), "using libonnxruntime");

    ort::init_from(path)
        .unwrap()
        .with_execution_providers([
            ort::ep::CUDAExecutionProvider::default().build(),
            ort::ep::CPUExecutionProvider::default().build(),
        ])
        .with_name("libsnout")
        .commit();
}
