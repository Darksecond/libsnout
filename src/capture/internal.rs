use image::GrayImage;
use v4l::video::capture::Parameters;
use v4l::{buffer::Type, io::traits::CaptureStream, prelude::UserptrStream, video::Capture};

use crate::capture::{CameraError, discovery::V4lSource};

#[derive(Copy, Clone, Debug)]
enum PixelFormat {
    Grey,
    Yuyv,
    Uyvy,
    Mjpeg,
}

pub struct V4lCamera {
    _device: v4l::Device,
    stream: UserptrStream,
    pixel_format: PixelFormat,
    pub width: usize,
    pub height: usize,
}

impl V4lCamera {
    pub fn open(source: V4lSource) -> Result<Self, CameraError> {
        let device = v4l::Device::new(source.index as usize)?;

        let format = v4l::Format::new(
            source.format.width,
            source.format.height,
            v4l::FourCC::new(&source.fourcc),
        );
        let format = device.set_format(&format)?;

        let params = Parameters::with_fps(source.format.fps);
        device.set_params(&params)?;

        let pixel_format = match &source.fourcc {
            b"GREY" => PixelFormat::Grey,
            b"YUYV" => PixelFormat::Yuyv,
            b"UYVY" => PixelFormat::Uyvy,
            b"MJPG" => PixelFormat::Mjpeg,
            _ => return Err(CameraError::OpenError),
        };

        let width = format.width as usize;
        let height = format.height as usize;

        let stream = UserptrStream::new(&device, Type::VideoCapture)?;

        Ok(Self {
            _device: device,
            stream,
            pixel_format,
            width,
            height,
        })
    }

    pub fn read_frame(&mut self, destination: &mut GrayImage) -> Result<(), CameraError> {
        let (buf, _meta) = self.stream.next()?;
        match self.pixel_format {
            PixelFormat::Grey => destination.copy_from_slice(buf),
            PixelFormat::Yuyv => {
                // extract Y channel: every other byte
                for (dst, &y) in destination.iter_mut().zip(buf.iter().step_by(2)) {
                    *dst = y;
                }
            }
            PixelFormat::Uyvy => {
                for (dst, &y) in destination.iter_mut().zip(buf[1..].iter().step_by(2)) {
                    *dst = y;
                }
            }
            PixelFormat::Mjpeg => {
                let img = image::load_from_memory(&buf[..])
                    .map_err(|e| CameraError::InvalidFrame(e.to_string()))?
                    .into_luma8();
                destination.copy_from_slice(img.as_raw());
            }
        }
        Ok(())
    }
}
