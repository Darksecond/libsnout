use std::path::Path;

use crate::{
    calibration::FaceShape,
    capture::Frame,
    pipeline::{
        FilterParameters, PipelineError,
        internal::{FrameToBurnTensor, inference::FaceInference, one_euro_filter::OneEuroFilter},
    },
};

pub struct FacePipeline {
    transfer: FrameToBurnTensor,
    inference: FaceInference,
    filter: OneEuroFilter,
}

impl FacePipeline {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let inference = FaceInference::new(path)?;

        Ok(Self {
            transfer: FrameToBurnTensor::new(1, 224, 224),
            inference,
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
            .transfer_frame(frame, &mut self.inference.input_tensor);

        let weights = self.inference.run()?;

        let filtered_weights = self.filter.filter(&weights);

        Ok(Some(filtered_weights))
    }
}
