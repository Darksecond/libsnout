use std::path::Path;

use burn::backend::wgpu::WgpuDevice;

use crate::{
    calibration::EyeShape,
    capture::Frame,
    nn::inference::EyeInference,
    pipeline::{
        FilterParameters, PipelineError,
        internal::{
            FrameToBurnTensor, eye_compositor::EyeCompositor, one_euro_filter::OneEuroFilter,
        },
    },
};

pub struct EyePipeline {
    device: WgpuDevice,
    transfer: FrameToBurnTensor,
    inference: EyeInference,
    collector: EyeCompositor,
    filter: OneEuroFilter,
}

impl EyePipeline {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let device = WgpuDevice::default();
        let inference = EyeInference::new(&device, path)?;

        Ok(Self {
            device,
            transfer: FrameToBurnTensor::new(8, 128, 128),
            inference,
            collector: EyeCompositor::new(),
            filter: OneEuroFilter::new(EyeShape::count()),
        })
    }

    pub fn filter(&self) -> FilterParameters {
        self.filter.parameters
    }

    pub fn set_filter(&mut self, parameters: FilterParameters) {
        self.filter.parameters = parameters;
    }

    pub fn run(&mut self, left: &Frame, right: &Frame) -> Result<Option<&[f32]>, PipelineError> {
        let Some(mat) = self.collector.compose(left, right) else {
            return Ok(None);
        };

        let tensor = self.transfer.transfer_composite(mat, &self.device);

        let weights = self.inference.run(tensor)?;

        let filtered_weights = self.filter.filter(&weights);

        Ok(Some(filtered_weights))
    }
}
