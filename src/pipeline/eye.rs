use std::path::Path;

use crate::{
    capture::Frame,
    pipeline::{Bounds, Filter, PipelineError, ShapeWeight},
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EyeShape {
    LeftEyePitch,
    LeftEyeYaw,
    LeftEyeLid,
    RightEyePitch,
    RightEyeYaw,
    RightEyeLid,
}

pub struct EyePipeline {
    // TODO
}

impl EyePipeline {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        let _ = path;
        todo!()
    }

    pub fn bounds(&self, shape: EyeShape) -> Bounds {
        let _ = shape;
        todo!()
    }

    pub fn set_bounds(&mut self, shape: EyeShape, bounds: Bounds) {
        let _ = (shape, bounds);
        todo!()
    }

    pub fn filter(&self) -> Filter {
        todo!()
    }

    pub fn set_filter(&mut self, filter: Filter) {
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
    ) -> Result<Option<&'a [ShapeWeight<EyeShape>]>, PipelineError> {
        let _ = (left, right);
        todo!()
    }
}
