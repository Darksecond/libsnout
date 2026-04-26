use std::path::Path;

use crate::{
    calibration::{FaceShape, ManualFaceCalibrator, Weights},
    capture::{
        CameraError, Frame, MonoCamera, discovery::CameraSource, processing::FramePreprocessor,
    },
    pipeline::FacePipeline,
    track::TrackerError,
};

pub struct FaceReport<'a> {
    pub raw_frame: &'a Frame,
    pub processed_frame: &'a Frame,
    pub weights: Weights<'a, FaceShape>,
}

pub struct FaceTracker {
    pub preprocessor: FramePreprocessor,
    pub pipeline: FacePipeline,
    pub calibrator: ManualFaceCalibrator,

    camera: Option<MonoCamera>,
    source: Option<CameraSource>,
}

impl FaceTracker {
    pub fn new(model: impl AsRef<Path>) -> Result<Self, TrackerError> {
        Ok(Self {
            camera: None,
            preprocessor: FramePreprocessor::new(),
            pipeline: FacePipeline::new(model)?,
            calibrator: ManualFaceCalibrator::new(),

            source: None,
        })
    }

    /// Sets the camera source for the tracker.
    ///
    /// If the source is different from the current source, the camera is reset.
    pub fn set_source(&mut self, source: Option<CameraSource>) {
        if self.source != source {
            self.camera = None;
        }

        self.source = source;
    }

    pub fn track(&mut self) -> Result<Option<FaceReport<'_>>, TrackerError> {
        match (self.camera.is_some(), self.source.is_some()) {
            (_, false) => {
                self.camera = None;
                return Ok(None);
            }
            (false, true) => {
                let source = self.source.unwrap();
                let camera =
                    MonoCamera::open(source).map_err(|e| TrackerError::Open(e.to_string()))?;
                self.camera = Some(camera);
            }
            (true, true) => {}
        }

        let camera = self.camera.as_mut().unwrap();

        let raw_frame = match camera.get_frame() {
            Ok(frame) => frame,
            Err(CameraError::InvalidFrame(_)) => {
                // TODO: Keep track of the amount of invalid frames
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };

        let processed_frame = self.preprocessor.process(raw_frame)?;

        let raw_weights = self.pipeline.run(processed_frame)?.unwrap();

        let weights = self.calibrator.calibrate(raw_weights);

        Ok(Some(FaceReport {
            raw_frame,
            processed_frame,
            weights,
        }))
    }
}
