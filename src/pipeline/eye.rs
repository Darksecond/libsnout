use std::path::Path;

use crate::{
    calibration::EyeShape,
    capture::Frame,
    pipeline::{
        FilterParameters, PipelineError,
        internal::{
            FrameToTensor, Inference, eye_compositor::EyeCompositor, one_euro_filter::OneEuroFilter,
        },
    },
};

pub struct EyePipeline {
    transfer: FrameToTensor,
    inference: Inference,
    collector: EyeCompositor,
    filter: OneEuroFilter,
}

impl EyePipeline {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let _ = path;

        Ok(Self {
            transfer: FrameToTensor::new(),
            inference: Inference::new(path).map_err(|e| PipelineError::Load(e.to_string()))?,
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

        self.transfer
            .transfer_composite(mat, &mut self.inference.input_tensor);

        let weights = self
            .inference
            .run()
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        let filtered_weights = self.filter.filter(&weights);

        Ok(Some(filtered_weights))
    }
}
