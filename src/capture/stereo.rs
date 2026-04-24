use image::GrayImage;

use crate::capture::{CameraError, Frame, discovery::CameraSource, internal::V4lCamera};

enum StereoCameraDevice {
    Single(V4lCamera, GrayImage),
    Dual(V4lCamera, V4lCamera),
}

pub struct StereoCamera {
    device: StereoCameraDevice,

    left_frame: Frame,
    right_frame: Frame,
}

impl StereoCamera {
    pub fn open(left: CameraSource, right: CameraSource) -> Result<Self, CameraError> {
        let left = V4lCamera::open(left)?;
        let right = V4lCamera::open(right)?;

        let width = left.width as u32;
        let height = left.height as u32;

        Ok(Self {
            device: StereoCameraDevice::Dual(left, right),
            left_frame: Frame::empty(width, height),
            right_frame: Frame::empty(width, height),
        })
    }

    pub fn open_sbs(source: CameraSource) -> Result<Self, CameraError> {
        let camera = V4lCamera::open(source)?;
        let sbs_buffer = GrayImage::new(camera.width as _, camera.height as _);

        let full_width = camera.width as u32;
        let height = camera.height as u32;
        let half_width = full_width / 2;

        Ok(Self {
            device: StereoCameraDevice::Single(camera, sbs_buffer),
            left_frame: Frame::empty(half_width, height),
            right_frame: Frame::empty(half_width, height),
        })
    }

    pub fn get_frames(&mut self) -> Result<(&Frame, &Frame), CameraError> {
        match &mut self.device {
            StereoCameraDevice::Single(camera, sbs_buffer) => {
                camera.read_frame(sbs_buffer)?;

                let full_width = sbs_buffer.width() as usize;
                let half_width = full_width / 2;
                let height = sbs_buffer.height() as usize;

                let source: &[u8] = sbs_buffer.as_ref();
                let left_destination: &mut [u8] = self.left_frame.image.as_mut();
                let right_destination: &mut [u8] = self.right_frame.image.as_mut();

                for y in 0..height {
                    let row = &source[y * full_width..(y + 1) * full_width];

                    left_destination[y * half_width..(y + 1) * half_width]
                        .copy_from_slice(&row[..half_width]);

                    right_destination[y * half_width..(y + 1) * half_width]
                        .copy_from_slice(&row[half_width..]);
                }
            }
            StereoCameraDevice::Dual(left, right) => {
                left.read_frame(&mut self.left_frame.image)?;
                right.read_frame(&mut self.right_frame.image)?;
            }
        };

        Ok((&self.left_frame, &self.right_frame))
    }
}
