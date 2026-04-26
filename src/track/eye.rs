use std::path::Path;

use crate::{
    calibration::{EyeCalibrator, EyeShape, Weights},
    capture::{
        CameraError, Frame, StereoCamera, discovery::CameraSource, processing::FramePreprocessor,
    },
    pipeline::EyePipeline,
    track::TrackerError,
};

pub struct EyeReport<'a> {
    pub left_raw_frame: &'a Frame,
    pub left_processed_frame: &'a Frame,
    pub right_raw_frame: &'a Frame,
    pub right_processed_frame: &'a Frame,
    pub weights: Weights<'a, EyeShape>,
}

pub struct EyeTracker {
    pub left_preprocessor: FramePreprocessor,
    pub right_preprocessor: FramePreprocessor,
    pub pipeline: EyePipeline,
    pub calibrator: EyeCalibrator,

    camera: Option<StereoCamera>,
    left_source: Option<CameraSource>,
    right_source: Option<CameraSource>,
}

impl EyeTracker {
    pub fn new(model: impl AsRef<Path>) -> Result<Self, TrackerError> {
        Ok(Self {
            left_preprocessor: FramePreprocessor::new(),
            right_preprocessor: FramePreprocessor::new(),
            pipeline: EyePipeline::new(model)?,
            calibrator: EyeCalibrator::new(),

            camera: None,
            left_source: None,
            right_source: None,
        })
    }

    /// Sets the camera source for the eye tracker.
    ///
    /// If the source has changed, the camera will be re-opened.
    /// If left equals right, the camera will be opened in side-by-side mode.
    /// If either source is `None`, the camera will not be opened.
    pub fn set_source(&mut self, left: Option<CameraSource>, right: Option<CameraSource>) {
        if self.left_source != left || self.right_source != right {
            self.camera = None;
        }

        self.left_source = left;
        self.right_source = right;
    }

    pub fn track(&mut self) -> Result<Option<EyeReport<'_>>, TrackerError> {
        if !self.ensure_camera()? {
            return Ok(None);
        }

        let camera = self.camera.as_mut().unwrap();

        let (left_raw_frame, right_raw_frame) = match camera.get_frames() {
            Ok(frames) => frames,
            Err(CameraError::InvalidFrame(_)) => {
                // TODO: Keep track of the amount of invalid frames
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        };

        let left_processed_frame = self.left_preprocessor.process(left_raw_frame)?;
        let right_processed_frame = self.right_preprocessor.process(right_raw_frame)?;

        let Ok(Some(raw_weights)) = self
            .pipeline
            .run(left_processed_frame, right_processed_frame)
        else {
            return Ok(None);
        };

        let weights = self.calibrator.calibrate(raw_weights);

        Ok(Some(EyeReport {
            left_raw_frame,
            right_raw_frame,
            left_processed_frame,
            right_processed_frame,
            weights,
        }))
    }

    fn ensure_camera(&mut self) -> Result<bool, TrackerError> {
        if self.camera.is_none() {
            let (Some(left), Some(right)) = (self.left_source, self.right_source) else {
                return Ok(false);
            };

            let camera = if left == right {
                StereoCamera::open_sbs(left)
            } else {
                StereoCamera::open(left, right)
            }
            .map_err(|e| TrackerError::Open(e.to_string()))?;

            self.camera = Some(camera);
        }

        Ok(true)
    }
}
