use std::path::Path;

use crate::{
    capture::Frame,
    pipeline::{
        Bounds, Filter, PipelineError, ShapeWeight,
        internal::{Inference, Transfer},
    },
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum FaceShape {
    CheekPuffLeft,
    CheekPuffRight,
    CheekSuckLeft,
    CheekSuckRight,
    JawOpen,
    JawForward,
    JawLeft,
    JawRight,
    NoseSneerLeft,
    NoseSneerRight,
    MouthFunnel,
    MouthPucker,
    MouthLeft,
    MouthRight,
    MouthRollUpper,
    MouthRollLower,
    MouthShrugUpper,
    MouthShrugLower,
    MouthClose,
    MouthSmileLeft,
    MouthSmileRight,
    MouthFrownLeft,
    MouthFrownRight,
    MouthDimpleLeft,
    MouthDimpleRight,
    MouthUpperUpLeft,
    MouthUpperUpRight,
    MouthLowerDownLeft,
    MouthLowerDownRight,
    MouthPressLeft,
    MouthPressRight,
    MouthStretchLeft,
    MouthStretchRight,
    TongueOut,
    TongueUp,
    TongueDown,
    TongueLeft,
    TongueRight,
    TongueRoll,
    TongueBendDown,
    TongueCurlUp,
    TongueSquish,
    TongueFlat,
    TongueTwistLeft,
    TongueTwistRight,
}

impl FaceShape {
    pub(crate) fn to_babble(self) -> &'static str {
        match self {
            FaceShape::CheekPuffLeft => "/cheekPuffLeft",
            FaceShape::CheekPuffRight => "/cheekPuffRight",
            FaceShape::CheekSuckLeft => "/cheekSuckLeft",
            FaceShape::CheekSuckRight => "/cheekSuckRight",
            FaceShape::JawOpen => "/jawOpen",
            FaceShape::JawForward => "/jawForward",
            FaceShape::JawLeft => "/jawLeft",
            FaceShape::JawRight => "/jawRight",
            FaceShape::NoseSneerLeft => "/noseSneerLeft",
            FaceShape::NoseSneerRight => "/noseSneerRight",
            FaceShape::MouthFunnel => "/mouthFunnel",
            FaceShape::MouthPucker => "/mouthPucker",
            FaceShape::MouthLeft => "/mouthLeft",
            FaceShape::MouthRight => "/mouthRight",
            FaceShape::MouthRollUpper => "/mouthRollUpper",
            FaceShape::MouthRollLower => "/mouthRollLower",
            FaceShape::MouthShrugUpper => "/mouthShrugUpper",
            FaceShape::MouthShrugLower => "/mouthShrugLower",
            FaceShape::MouthClose => "/mouthClose",
            FaceShape::MouthSmileLeft => "/mouthSmileLeft",
            FaceShape::MouthSmileRight => "/mouthSmileRight",
            FaceShape::MouthFrownLeft => "/mouthFrownLeft",
            FaceShape::MouthFrownRight => "/mouthFrownRight",
            FaceShape::MouthDimpleLeft => "/mouthDimpleLeft",
            FaceShape::MouthDimpleRight => "/mouthDimpleRight",
            FaceShape::MouthUpperUpLeft => "/mouthUpperUpLeft",
            FaceShape::MouthUpperUpRight => "/mouthUpperUpRight",
            FaceShape::MouthLowerDownLeft => "/mouthLowerDownLeft",
            FaceShape::MouthLowerDownRight => "/mouthLowerDownRight",
            FaceShape::MouthPressLeft => "/mouthPressLeft",
            FaceShape::MouthPressRight => "/mouthPressRight",
            FaceShape::MouthStretchLeft => "/mouthStretchLeft",
            FaceShape::MouthStretchRight => "/mouthStretchRight",
            FaceShape::TongueOut => "/tongueOut",
            FaceShape::TongueUp => "/tongueUp",
            FaceShape::TongueDown => "/tongueDown",
            FaceShape::TongueLeft => "/tongueLeft",
            FaceShape::TongueRight => "/tongueRight",
            FaceShape::TongueRoll => "/tongueRoll",
            FaceShape::TongueBendDown => "/tongueBendDown",
            FaceShape::TongueCurlUp => "/tongueCurlUp",
            FaceShape::TongueSquish => "/tongueSquish",
            FaceShape::TongueFlat => "/tongueFlat",
            FaceShape::TongueTwistLeft => "/tongueTwistLeft",
            FaceShape::TongueTwistRight => "/tongueTwistRight",
        }
    }
}

impl FaceShape {
    const fn from(value: usize) -> Self {
        unsafe { std::mem::transmute(value as u8) }
    }

    const fn count() -> usize {
        Self::TongueTwistRight as usize + 1
    }
}

pub struct FacePipeline {
    transfer: Transfer,
    bounds: Vec<Bounds>,
    inference: Inference,
    weights: Vec<ShapeWeight<FaceShape>>,
    // TODO filter
}

impl FacePipeline {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PipelineError> {
        Ok(Self {
            bounds: vec![Bounds::new_01(); FaceShape::count()],
            transfer: Transfer::new(),
            inference: Inference::new(path).map_err(|_| PipelineError::UnknownModel)?,
            weights: default_weights(),
        })
    }

    pub fn bounds(&self, shape: FaceShape) -> Bounds {
        self.bounds[shape as usize]
    }

    pub fn set_bounds(&mut self, shape: FaceShape, bounds: Bounds) {
        self.bounds[shape as usize] = bounds;
    }

    pub fn filter(&self) -> Filter {
        todo!()
    }

    pub fn set_filter(&mut self, filter: Filter) {
        let _ = filter;
        todo!()
    }

    pub fn run<'a>(
        &'a mut self,
        frame: &Frame,
    ) -> Result<Option<&'a [ShapeWeight<FaceShape>]>, PipelineError> {
        self.transfer
            .transfer(&frame.mat, &mut self.inference.input_tensor);

        let weights = self
            .inference
            .run()
            .map_err(|_| PipelineError::InferenceError)?;

        // TODO: Filtering
        // Update weights with new values
        for (weight, value) in self.weights.iter_mut().zip(weights) {
            weight.raw = value;
            weight.value = value;
        }

        Ok(Some(self.weights.as_slice()))
    }
}

fn default_weights() -> Vec<ShapeWeight<FaceShape>> {
    let mut weights = Vec::with_capacity(FaceShape::count());

    for index in 0..FaceShape::count() {
        weights.push(ShapeWeight {
            shape: FaceShape::from(index),
            raw: 0.0,
            value: 0.0,
        })
    }

    weights
}
