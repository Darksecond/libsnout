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
    inference: Option<FaceInference>,
    filter: OneEuroFilter,
}

impl FacePipeline {
    pub fn new() -> Self {
        Self {
            transfer: FrameToBurnTensor::new(1, 224, 224),
            inference: None,
            filter: OneEuroFilter::new(FaceShape::count()),
        }
    }

    pub fn set_model(&mut self, path: Option<impl AsRef<Path>>) -> Result<(), PipelineError> {
        if let Some(path) = path {
            let inference = FaceInference::new(path)?;
            self.inference = Some(inference);
        } else {
            self.inference = None;
        }

        Ok(())
    }

    pub fn filter(&self) -> FilterParameters {
        self.filter.parameters
    }

    pub fn set_filter(&mut self, parameters: FilterParameters) {
        self.filter.parameters = parameters;
    }

    pub fn run(&mut self, frame: &Frame) -> Result<Option<&[f32]>, PipelineError> {
        let Some(inference) = self.inference.as_mut() else {
            return Ok(None);
        };

        self.transfer
            .transfer_frame(frame, &mut inference.input_tensor);

        let weights = inference.run()?;

        let filtered_weights = self.filter.filter(&weights);

        Ok(Some(filtered_weights))
    }
}
