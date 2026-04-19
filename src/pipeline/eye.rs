use std::path::Path;

use crate::{
    capture::Frame,
    pipeline::{FilterParameters, PipelineError, PipelineWeights},
};

pub struct EyePipeline {
    // TODO
}

impl EyePipeline {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let _ = path;
        todo!()
    }

    pub fn filter(&self) -> FilterParameters {
        todo!()
    }

    pub fn set_filter(&mut self, filter: FilterParameters) {
        let _ = filter;
        todo!()
    }

    pub fn link_eyes(&self) -> bool {
        todo!()
    }

    pub fn set_link_eyes(&mut self, value: bool) {
        let _ = value;
        todo!()
    }

    pub fn run<'a>(
        &'a mut self,
        left: &Frame,
        right: &Frame,
    ) -> Result<Option<PipelineWeights<'a>>, PipelineError> {
        let _ = (left, right);
        todo!()
    }
}
