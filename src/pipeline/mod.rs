pub mod eye;
pub mod face;

mod internal;

use thiserror::Error;

/// Initialize the ONNX runtime.
pub fn init_runtime() {
    ort::init()
        .with_execution_providers([ort::ep::CUDA::default().build()])
        .with_name("libsnout")
        .commit();
}

#[derive(Clone, Debug, Error)]
pub enum PipelineError {
    #[error("Failed to load model: {0}")]
    Load(String),
    #[error("Inference failed: {0}")]
    Inference(String),
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct FilterParameters {
    pub enable: bool,
    pub min_cutoff: f32,
    pub beta: f32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PipelineWeights<'a> {
    pub raw: &'a [f32],
    pub filtered: &'a [f32],
}
