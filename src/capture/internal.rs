use image::GrayImage;
use opencv::{
    core::{AlgorithmHint, Mat, MatTraitConst, MatTraitConstManual},
    imgproc::COLOR_BGR2GRAY,
    videoio::{
        CAP_ANY, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture, VideoCaptureTrait,
        VideoCaptureTraitConst,
    },
};

use crate::capture::{CameraError, discovery::CameraSource};

pub struct OpenCvCamera {
    device: VideoCapture,
    raw: Mat,
    gray: Mat,
    pub width: usize,
    pub height: usize,
}

impl OpenCvCamera {
    pub fn open(source: CameraSource) -> Result<Self, CameraError> {
        let index = match source {
            CameraSource::Index(i) => i,
        };

        let device = VideoCapture::new(index as _, CAP_ANY)
            .map_err(|e| CameraError::Internal(e.to_string()))?;

        let width = device
            .get(CAP_PROP_FRAME_WIDTH)
            .map_err(|e| CameraError::Internal(e.to_string()))? as usize;
        let height = device
            .get(CAP_PROP_FRAME_HEIGHT)
            .map_err(|e| CameraError::Internal(e.to_string()))? as usize;

        Ok(Self {
            device,
            raw: Mat::default(),
            gray: Mat::default(),
            width,
            height,
        })
    }

    pub fn read_frame(&mut self, destination: &mut GrayImage) -> Result<(), CameraError> {
        self.device.read(&mut self.raw)?;

        if self.raw.empty() {
            return Err(CameraError::InvalidFrame);
        }

        let gray = if self.raw.channels() == 1 {
            &self.raw
        } else {
            opencv::imgproc::cvt_color(
                &self.raw,
                &mut &mut self.gray,
                COLOR_BGR2GRAY,
                0,
                AlgorithmHint::ALGO_HINT_DEFAULT,
            )?;

            &self.gray
        };

        let src = gray.data_typed::<u8>()?;
        destination.copy_from_slice(src);

        Ok(())
    }
}
