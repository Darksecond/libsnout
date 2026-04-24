use crate::{
    models::{dual_eye_net::DualEyeNet, face_net::FaceNet},
    pipeline::PipelineError,
};
use std::path::Path;

use burn::{
    Tensor,
    backend::{Wgpu, wgpu::WgpuDevice},
};
use burn_store::{BurnpackStore, ModuleSnapshot};

pub struct FaceInference {
    model: FaceNet<Wgpu>,
    output: Vec<f32>,
}

impl FaceInference {
    pub fn new(device: &WgpuDevice, path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let model = {
            let mut model = FaceNet::new(device);
            let mut store = BurnpackStore::from_file(path);
            model
                .load_from(&mut store)
                .map_err(|e| PipelineError::Load(e.to_string()))?;
            model
        };

        Ok(Self {
            model,
            output: vec![0.; 45],
        })
    }

    pub fn run(&mut self, tensor: Tensor<Wgpu, 4>) -> Result<&[f32], PipelineError> {
        let output = self.model.forward(tensor);
        let data = output.into_data();
        let src = data
            .as_slice()
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        self.output.copy_from_slice(src);

        Ok(&self.output)
    }
}

pub struct EyeInference {
    model: DualEyeNet<Wgpu>,
    output: Vec<f32>,
}

impl EyeInference {
    pub fn new(device: &WgpuDevice, path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let model = DualEyeNet::load_safetensors(path, device)
            .map_err(|e| PipelineError::Load(e.to_string()))?;

        Ok(Self {
            model,
            output: vec![0.; 6],
        })
    }

    pub fn run(&mut self, tensor: Tensor<Wgpu, 4>) -> Result<&[f32], PipelineError> {
        let output = self.model.forward(tensor);
        let data = output.into_data();
        let src = data
            .as_slice()
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        self.output.copy_from_slice(src);

        Ok(&self.output)
    }
}
