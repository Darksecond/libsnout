use opencv::core::Mat;

use crate::capture::{
    Calibration, CameraError, Frame,
    discovery::CameraSource,
    internal::{Calibrator, Camera},
};

pub struct MonoCamera {
    calibrator: Calibrator,
    inner: Camera,
    frame: Frame,
}

impl MonoCamera {
    pub fn open(source: CameraSource) -> Result<Self, CameraError> {
        Ok(Self {
            calibrator: Calibrator::new(Calibration::default()),
            inner: Camera::open(source).map_err(|_| CameraError::OpenError)?,
            frame: unsafe { Frame::new_unchecked(Mat::default()) },
        })
    }

    pub fn calibration(&self) -> Calibration {
        self.calibrator.calibration
    }

    pub fn set_calibration(&mut self, calibration: Calibration) {
        self.calibrator.calibration = calibration;
    }

    pub fn get_frame(&mut self) -> Result<&Frame, CameraError> {
        match self.inner.get_frame() {
            Ok(Some(mat)) => {
                self.calibrator
                    .calibrate(mat, &mut self.frame.mat)
                    .map_err(|e| CameraError::Internal(e.message))?;

                Ok(&self.frame)
            }
            Ok(None) => Err(CameraError::InvalidFrame),
            Err(_) => Err(CameraError::Disconnected),
        }
    }
}
