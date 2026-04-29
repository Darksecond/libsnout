use crate::{
    calibration::{FaceShape, ManualFaceCalibrator, Weights},
    capture::{
        CameraError, Frame, MonoCamera,
        discovery::{CameraInfo, CameraSource},
        processing::FramePreprocessor,
    },
    config::Config,
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
    pub fn new() -> Self {
        Self {
            preprocessor: FramePreprocessor::new(),
            pipeline: FacePipeline::new(),
            calibrator: ManualFaceCalibrator::new(),

            camera: None,
            source: None,
        }
    }

    pub fn with_config(cameras: &[CameraInfo], config: &Config) -> Result<Self, TrackerError> {
        let mut tracker = Self::new();

        tracker.pipeline.set_model(config.face.model.as_ref())?;

        let camera = cameras
            .iter()
            .find(|s| s.display_name() == config.face.camera)
            .map(|c| c.source);

        tracker.set_source(camera);

        if let Some(crop) = &config.face.crop {
            tracker.preprocessor.set_crop(*crop);
        }

        if let Some(transform) = &config.face.transform {
            tracker.preprocessor.set_config(*transform);
        }

        Ok(tracker)
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
        if !self.ensure_camera()? {
            return Ok(None);
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

    fn ensure_camera(&mut self) -> Result<bool, TrackerError> {
        if self.camera.is_none() {
            let Some(source) = self.source else {
                return Ok(false);
            };

            self.camera =
                Some(MonoCamera::open(source).map_err(|e| TrackerError::Open(e.to_string()))?);
        }

        Ok(true)
    }
}
