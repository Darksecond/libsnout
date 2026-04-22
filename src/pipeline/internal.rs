pub mod eye_compositor;
pub mod one_euro_filter;

use crate::{capture::Frame, pipeline::internal::eye_compositor::CompositeImage};

pub struct FrameToBurnTensor {
    channels: usize,
    height: usize,
    width: usize,
    buffer: Vec<f32>,
}

impl FrameToBurnTensor {
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            height,
            width,
            buffer: vec![0.0f32; channels * height * width],
        }
    }

    pub fn transfer_frame(
        &mut self,
        source: &Frame,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> burn::Tensor<burn::backend::Wgpu, 4> {
        self.transfer(
            &source.image.as_ref(),
            source.image.width(),
            source.image.height(),
            1,
            device,
        )
    }

    pub fn transfer_composite(
        &mut self,
        source: &CompositeImage,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> burn::Tensor<burn::backend::Wgpu, 4> {
        self.transfer(
            source.data.as_ref(),
            source.width,
            source.height,
            source.channels,
            device,
        )
    }

    fn transfer(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> burn::Tensor<burn::backend::Wgpu, 4> {
        let (tc, th, tw) = (self.channels, self.height, self.width);
        assert_eq!(channels, tc, "channel count mismatch");

        let pixels_per_channel = (width as usize) * (height as usize);

        for c in 0..tc {
            let ch_data = &data[c * pixels_per_channel..(c + 1) * pixels_per_channel];
            let source: image::ImageBuffer<image::Luma<u8>, &[u8]> =
                image::ImageBuffer::from_raw(width, height, ch_data).unwrap();

            let resized_buf = image::imageops::resize(
                &source,
                tw as u32,
                th as u32,
                image::imageops::FilterType::Triangle,
            );

            for (i, &pixel) in resized_buf.as_raw().iter().enumerate() {
                self.buffer[c * th * tw + i] = pixel as f32 / 255.0;
            }
        }

        burn::Tensor::from_data(
            burn::tensor::TensorData::new(self.buffer.clone(), [1, tc, th, tw]),
            device,
        )
    }
}
