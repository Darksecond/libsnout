use crate::pipeline::PipelineError;
use std::path::Path;

use ndarray::Array4;
use ort::{
    inputs,
    session::{Session, builder::SessionBuilder},
};

pub struct FaceInference {
    session: Session,
    input_name: String,
    pub input_tensor: ort::value::Tensor<f32>,
    output: Vec<f32>,
}

impl FaceInference {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let session = builder()
            .map_err(|e| PipelineError::Load(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| PipelineError::Load(e.to_string()))?;

        let input0 = &session.inputs()[0];
        let input_name = input0.name().to_string();

        let dims = input0
            .dtype()
            .tensor_shape()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let input_tensor = ort::value::Tensor::from_array(Array4::<f32>::zeros((
            1,
            dims[1] as _,
            dims[2] as _,
            dims[3] as _,
        )))
        .map_err(|e| PipelineError::Load(e.to_string()))?;

        Ok(Self {
            session,
            input_name,
            input_tensor,
            output: vec![0.; 45],
        })
    }

    pub fn run(&mut self) -> Result<&[f32], PipelineError> {
        let outputs = self
            .session
            .run(inputs![&self.input_name => &self.input_tensor])
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        let blendshapes = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        self.output.copy_from_slice(blendshapes.1);

        Ok(&self.output)
    }
}

pub struct EyeInference {
    session: Session,
    input_name: String,
    pub input_tensor: ort::value::Tensor<f32>,
    output: Vec<f32>,
}

impl EyeInference {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let session = builder()
            .map_err(|e| PipelineError::Load(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| PipelineError::Load(e.to_string()))?;

        let input0 = &session.inputs()[0];
        let input_name = input0.name().to_string();

        let dims = input0
            .dtype()
            .tensor_shape()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let input_tensor = ort::value::Tensor::from_array(Array4::<f32>::zeros((
            1,
            dims[1] as _,
            dims[2] as _,
            dims[3] as _,
        )))
        .map_err(|e| PipelineError::Load(e.to_string()))?;

        Ok(Self {
            session,
            input_name,
            input_tensor,
            output: vec![0.; 6],
        })
    }

    pub fn run(&mut self) -> Result<&[f32], PipelineError> {
        let outputs = self
            .session
            .run(inputs![&self.input_name => &self.input_tensor])
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        let blendshapes = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        self.output.copy_from_slice(blendshapes.1);

        Ok(&self.output)
    }
}

fn builder() -> Result<SessionBuilder, ort::Error> {
    let builder = Session::builder()?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_intra_op_spinning(false)?
        .with_inter_op_spinning(false)?
        .with_memory_pattern(true)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)?;

    Ok(builder)
}
