pub mod eye_compositor;
pub mod one_euro_filter;

use std::path::Path;

use ndarray::Array4;
use ort::{
    Error, inputs,
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

use crate::{capture::Frame, pipeline::internal::eye_compositor::CompositeImage};

pub struct Inference {
    session: Session,
    input_name: String,
    pub input_tensor: Tensor<f32>,
    pub output: Vec<f32>,
}

impl Inference {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, Error> {
        let session = builder()?.commit_from_file(path)?;

        let input0 = &session.inputs()[0];
        let input_name = input0.name().to_string();

        let dims = input0
            .dtype()
            .tensor_shape()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let output_dims = session.outputs()[0]
            .dtype()
            .tensor_shape()
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>();

        let input_tensor = Tensor::from_array(Array4::<f32>::zeros((
            1,
            dims[1] as _,
            dims[2] as _,
            dims[3] as _,
        )))?;

        Ok(Self {
            session,
            input_tensor,
            input_name,
            output: vec![0.; output_dims[1] as usize],
        })
    }

    pub fn run(&mut self) -> Result<&[f32], Error> {
        let outputs = self
            .session
            .run(inputs![&self.input_name => &self.input_tensor])?;

        let blendshapes = outputs[0].try_extract_tensor::<f32>()?;

        self.output.copy_from_slice(blendshapes.1);

        Ok(&self.output)
    }
}

fn builder() -> Result<SessionBuilder, Error> {
    let builder = Session::builder()?
        .with_inter_threads(1)?
        .with_intra_threads(1)?
        .with_intra_op_spinning(false)?
        .with_inter_op_spinning(false)?
        .with_memory_pattern(true)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)?;

    Ok(builder)
}

pub struct FrameToTensor {}

impl FrameToTensor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn transfer_frame(&mut self, source: &Frame, destination: &mut Tensor<f32>) {
        self.transfer(
            &source.image.as_ref(),
            source.image.width(),
            source.image.height(),
            1,
            destination,
        );
    }

    pub fn transfer_composite(&mut self, source: &CompositeImage, destination: &mut Tensor<f32>) {
        self.transfer(
            source.data.as_ref(),
            source.width,
            source.height,
            source.channels,
            destination,
        );
    }

    pub fn transfer(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        destination: &mut Tensor<f32>,
    ) {
        let dims = destination.shape();

        let tc = dims[1] as usize;
        let th = dims[2] as usize;
        let tw = dims[3] as usize;

        assert_eq!(channels, tc, "channel count mismatch");

        let pixels_per_channel = (width as usize) * (height as usize);
        let mut out = destination.extract_array_mut();

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
                out[[0, c, i / tw as usize, i % tw as usize]] = pixel as f32 / 255.0;
            }
        }
    }
}
