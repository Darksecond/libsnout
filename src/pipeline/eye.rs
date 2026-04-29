use std::path::Path;

use crate::{
    calibration::EyeShape,
    capture::Frame,
    pipeline::{
        FilterParameters, PipelineError,
        internal::{
            FrameToBurnTensor, eye_compositor::EyeCompositor, inference::EyeInference,
            one_euro_filter::OneEuroFilter,
        },
    },
};

pub struct EyePipeline {
    transfer: FrameToBurnTensor,
    inference: Option<EyeInference>,
    collector: EyeCompositor,
    filter: OneEuroFilter,
}

impl EyePipeline {
    pub fn new() -> Self {
        Self {
            transfer: FrameToBurnTensor::new(8, 128, 128),
            inference: None,
            collector: EyeCompositor::new(),
            filter: OneEuroFilter::new(EyeShape::count()),
        }
    }

    pub fn set_model(&mut self, path: Option<impl AsRef<Path>>) -> Result<(), PipelineError> {
        if let Some(path) = path {
            self.inference = Some(EyeInference::new(path)?);
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

    pub fn run(&mut self, left: &Frame, right: &Frame) -> Result<Option<&[f32]>, PipelineError> {
        let Some(inference) = self.inference.as_mut() else {
            return Ok(None);
        };

        let Some(mat) = self.collector.compose(left, right) else {
            return Ok(None);
        };

        self.transfer
            .transfer_composite(mat, &mut inference.input_tensor);

        let weights = inference.run()?;

        let filtered_weights = self.filter.filter(&weights);

        Ok(Some(filtered_weights))
    }
}
