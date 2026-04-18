use opencv::{
    boxed_ref::BoxedRef,
    core::{Mat, MatTraitConst, Rect},
};

use crate::capture::{CameraError, Frame, discovery::CameraSource, internal::Camera};

enum StereoCameraDevice {
    Single(Camera),
    Dual(Camera, Camera),
}

pub struct StereoCamera {
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
            left_frame: unsafe { Frame::new_unchecked(Mat::default()) },
            right_frame: unsafe { Frame::new_unchecked(Mat::default()) },
        })
    }

    pub fn get_frames(&mut self) -> Result<(&Frame, &Frame), CameraError> {
        match &mut self.device {
            StereoCameraDevice::Single(camera) => {
                let mat = camera.require_frame()?;
                let (left, right) =
                    split(&mat).map_err(|e| CameraError::Internal(e.to_string()))?;

                left.copy_to(&mut self.left_frame.mat)
                    .map_err(|e| CameraError::Internal(e.to_string()))?;

                right
                    .copy_to(&mut self.right_frame.mat)
                    .map_err(|e| CameraError::Internal(e.to_string()))?;
            }
            StereoCameraDevice::Dual(left, right) => {
                let left = left.require_frame()?;
                let right = right.require_frame()?;

                left.copy_to(&mut self.left_frame.mat)
                    .map_err(|e| CameraError::Internal(e.to_string()))?;

                right
                    .copy_to(&mut self.right_frame.mat)
                    .map_err(|e| CameraError::Internal(e.to_string()))?;
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
