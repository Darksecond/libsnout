use std::path::Path;

use crate::{
    calibration::face::FaceShape,
    capture::Frame,
    pipeline::{
        FilterParameters, PipelineError,
        internal::{Inference, Transfer, one_euro_filter::OneEuroFilter},
    },
};

pub struct FacePipeline {
    transfer: Transfer,
    inference: Inference,
    filter: OneEuroFilter,
}

impl FacePipeline {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        Ok(Self {
            transfer: Transfer::new(),
            inference: Inference::new(path).map_err(|e| PipelineError::Load(e.to_string()))?,
            filter: OneEuroFilter::new(FaceShape::count()),
        })
    }

    pub fn filter(&self) -> FilterParameters {
        self.filter.parameters
    }

    pub fn set_filter(&mut self, parameters: FilterParameters) {
        self.filter.parameters = parameters;
    }

    pub fn run(&mut self, frame: &Frame) -> Result<Option<&[f32]>, PipelineError> {
        self.transfer
            .transfer(&frame.mat, &mut self.inference.input_tensor);

        let weights = self
            .inference
            .run()
            .map_err(|e| PipelineError::Inference(e.to_string()))?;

        let filtered_weights = self.filter.filter(&weights);

        Ok(Some(filtered_weights))
    }
}
