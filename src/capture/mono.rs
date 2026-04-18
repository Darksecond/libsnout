use opencv::core::{Mat, MatTraitConst};

use crate::capture::{CameraError, Frame, discovery::CameraSource, internal::Camera};

pub struct MonoCamera {
    inner: Camera,
    frame: Frame,
}

impl MonoCamera {
    pub fn open(source: CameraSource) -> Result<Self, CameraError> {
        Ok(Self {
            inner: Camera::open(source).map_err(|_| CameraError::OpenError)?,
            frame: unsafe { Frame::new_unchecked(Mat::default()) },
        })
    }

    pub fn get_frame(&mut self) -> Result<&Frame, CameraError> {
        let mat = self.inner.require_frame()?;

        mat.copy_to(&mut self.frame.mat)
            .map_err(|e| CameraError::Internal(e.to_string()))?;

        Ok(&self.frame)
    }
}
