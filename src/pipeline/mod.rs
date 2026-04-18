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
pub struct ShapeWeight<S> {
    pub shape: S,
    pub raw: f32,
    pub value: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Bounds {
    pub min: f32,
    pub max: f32,
    pub lower: f32,
    pub upper: f32,
}

impl Bounds {
    pub(crate) const fn new_01() -> Self {
        Self {
            min: 0.,
            max: 1.,
            lower: 0.,
            upper: 1.,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Filter {
    pub enable: bool,
    pub min_cutoff: f32,
    pub beta: f32,
}
