use crate::capture::{CameraError, Frame, discovery::CameraSource, internal::OpenCvCamera};

pub struct MonoCamera {
    inner: OpenCvCamera,
    frame: Frame,
}

impl MonoCamera {
    pub fn open(source: CameraSource) -> Result<Self, CameraError> {
        let inner = OpenCvCamera::open(source)?;

        Ok(Self {
            frame: Frame::empty(inner.width as u32, inner.height as u32),
            inner,
        })
    }

    pub fn get_frame(&mut self) -> Result<&Frame, CameraError> {
        self.inner.read_frame(&mut self.frame.image)?;

        Ok(&self.frame)
    }
}
