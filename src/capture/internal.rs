use opencv::{
    core::{AlgorithmHint, Mat, MatTraitConst},
    imgproc::{self, COLOR_BGR2GRAY},
    videoio::{CAP_ANY, VideoCapture, VideoCaptureTrait},
};

use crate::capture::{Calibration, CameraError, discovery::CameraSource};

pub struct Camera {
    device: VideoCapture,
    raw: Mat,
    gray: Mat,
}

impl Camera {
    pub fn open(source: CameraSource) -> Result<Self, opencv::Error> {
        let index = match source {
            CameraSource::Index(i) => i,
        };

        Ok(Self {
            device: VideoCapture::new(index as _, CAP_ANY)?,
            raw: Mat::default(),
            gray: Mat::default(),
        })
    }

    pub fn get_frame(&mut self) -> Result<Option<&Mat>, opencv::Error> {
        self.device.read(&mut self.raw)?;

        if self.raw.empty() {
            return Ok(None);
        }

        if self.raw.channels() == 1 {
            return Ok(Some(&self.raw));
        }

        // Turn gray-scale
        imgproc::cvt_color(
            &self.raw,
            &mut &mut self.gray,
            COLOR_BGR2GRAY,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        Ok(Some(&self.gray))
    }

    pub fn require_frame(&mut self) -> Result<&Mat, CameraError> {
        match self.get_frame() {
            Ok(Some(mat)) => Ok(mat),
            Ok(None) => Err(CameraError::InvalidFrame),
            Err(e) => Err(CameraError::Internal(e.to_string())),
        }
    }
}

pub struct Calibrator {
    pub calibration: Calibration,
}

impl Calibrator {
    pub fn new(calibration: Calibration) -> Self {
        Self { calibration }
    }

    pub fn calibrate(
        &mut self,
        source: &impl MatTraitConst,
        destination: &mut Mat,
    ) -> Result<(), opencv::Error> {
        // TODO: Other things

        // TODO: Double check, maybe `1. - brightness`
        source.convert_to(destination, source.typ(), self.calibration.brightness, 0.)?;

        Ok(())
    }
}
