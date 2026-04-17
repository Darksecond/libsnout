use opencv::{
    boxed_ref::BoxedRef,
    core::{Mat, MatTraitConst, Rect},
};

use crate::capture::{
    Calibration, CameraError, Frame,
    discovery::CameraSource,
    internal::{Calibrator, Camera},
};

enum StereoCameraDevice {
    Single(Camera),
    Dual(Camera, Camera),
}

pub struct StereoCamera {
    left_calibrator: Calibrator,
    right_calibrator: Calibrator,

    device: StereoCameraDevice,

    left_frame: Frame,
    right_frame: Frame,
}

impl StereoCamera {
    pub fn open(left: CameraSource, right: CameraSource) -> Result<Self, CameraError> {
        let left = Camera::open(left).map_err(|_| CameraError::OpenError)?;
        let right = Camera::open(right).map_err(|_| CameraError::OpenError)?;

        Self::with_device(StereoCameraDevice::Dual(left, right))
    }

    pub fn open_sbs(source: CameraSource) -> Result<Self, CameraError> {
        let camera = Camera::open(source).map_err(|_| CameraError::OpenError)?;

        Self::with_device(StereoCameraDevice::Single(camera))
    }

    fn with_device(device: StereoCameraDevice) -> Result<Self, CameraError> {
        Ok(Self {
            device,
            left_calibrator: Calibrator::new(Calibration {
                gamma: 0.66,
                ..Calibration::default()
            }),
            right_calibrator: Calibrator::new(Calibration {
                gamma: 0.66,
                ..Calibration::default()
            }),
            left_frame: unsafe { Frame::new_unchecked(Mat::default()) },
            right_frame: unsafe { Frame::new_unchecked(Mat::default()) },
        })
    }

    pub fn left_calibration(&self) -> Calibration {
        self.left_calibrator.calibration
    }

    pub fn set_left_calibration(&mut self, calibration: Calibration) {
        self.left_calibrator.calibration = calibration;
    }

    pub fn right_calibration(&self) -> Calibration {
        self.right_calibrator.calibration
    }

    pub fn set_right_calibration(&mut self, calibration: Calibration) {
        self.right_calibrator.calibration = calibration;
    }

    pub fn get_frames(&mut self) -> Result<(&Frame, &Frame), CameraError> {
        match &mut self.device {
            StereoCameraDevice::Single(camera) => {
                match camera.get_frame() {
                    Ok(Some(mat)) => {
                        let (left, right) =
                            split(&mat).map_err(|e| CameraError::Internal(e.to_string()))?;

                        // TODO: Calibrator
                        self.left_calibrator
                            .calibrate(&left, &mut self.left_frame.mat)
                            .map_err(|e| CameraError::Internal(e.to_string()))?;

                        // TODO: Calibrator
                        self.right_calibrator
                            .calibrate(&right, &mut self.right_frame.mat)
                            .map_err(|e| CameraError::Internal(e.to_string()))?;

                        Ok(())
                    }
                    Ok(None) => Err(CameraError::InvalidFrame),
                    Err(_) => Err(CameraError::Disconnected),
                }?;
            }
            StereoCameraDevice::Dual(left, right) => {
                match left.get_frame() {
                    Ok(Some(mat)) => {
                        // TODO: Calibrator
                        mat.copy_to(&mut self.left_frame.mat)
                            .map_err(|e| CameraError::Internal(e.to_string()))?;

                        Ok(())
                    }
                    Ok(None) => Err(CameraError::InvalidFrame),
                    Err(_) => Err(CameraError::Disconnected),
                }?;

                match right.get_frame() {
                    Ok(Some(mat)) => {
                        // TODO: Calibrator
                        mat.copy_to(&mut self.right_frame.mat)
                            .map_err(|e| CameraError::Internal(e.to_string()))?;

                        Ok(())
                    }
                    Ok(None) => Err(CameraError::InvalidFrame),
                    Err(_) => Err(CameraError::Disconnected),
                }?;
            }
        };

        Ok((&self.left_frame, &self.right_frame))
    }
}

pub fn split<'a>(mat: &'a Mat) -> Result<(BoxedRef<'a, Mat>, BoxedRef<'a, Mat>), opencv::Error> {
    let width = mat.cols() as i32;
    let height = mat.rows() as i32;

    let left = mat.roi(Rect::new(0, 0, width / 2, height))?;
    let right = mat.roi(Rect::new(width / 2, 0, width / 2, height))?;

    Ok((left, right))
}
