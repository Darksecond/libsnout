use image::GrayImage;
use v4l::{buffer::Type, io::traits::CaptureStream, prelude::UserptrStream, video::Capture};

use crate::capture::{CameraError, discovery::CameraSource};

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
    pub fn open(source: CameraSource) -> Result<Self, CameraError> {
        let source = match source {
            CameraSource::Index(i) => i,
        };

        let device = v4l::Device::new(source as _)?;
        let mut format = device.format()?;

        let preferred = [
            (v4l::FourCC::new(b"GREY"), PixelFormat::Grey),
            (v4l::FourCC::new(b"YUYV"), PixelFormat::Yuyv),
            (v4l::FourCC::new(b"UYVY"), PixelFormat::Uyvy),
            (v4l::FourCC::new(b"MJPG"), PixelFormat::Mjpeg),
        ];

        // TODO: Revisit this and improve.
        let mut pixel_format = None;
        for (fourcc, pf) in preferred {
            format.fourcc = fourcc;
            let actual = device.set_format(&format)?;
            if actual.fourcc == fourcc {
                format = actual;
                pixel_format = Some(pf);
                break;
            }
        }

        let pixel_format = pixel_format.ok_or(CameraError::OpenError)?;
        let width = format.width as usize;
        let height = format.height as usize;

        dbg!(pixel_format);

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
