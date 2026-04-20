pub mod eye_compositor;
pub mod one_euro_filter;

use std::path::Path;

use ndarray::Array4;
use opencv::core::{Mat, MatTraitConst};
use ort::{
    Error, inputs,
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

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

pub struct FrameToTensor {
    f32_mat: Mat,
    resized: Mat,
}

impl FrameToTensor {
    pub fn new() -> Self {
        Self {
            f32_mat: Mat::default(),
            resized: Mat::default(),
        }
    }

    /// Transfer from HWC to NCHW
    pub fn transfer(&mut self, source: &Mat, destination: &mut Tensor<f32>) {
        let dimensions = destination.shape();

        assert_eq!(
            source.channels(),
            dimensions[1] as _,
            "channel count mismatch"
        );

        // Convert mat to f32 (and divide by 255)
        source
            .convert_to(
                &mut self.f32_mat,
                opencv::core::CV_32FC(source.channels()),
                1.0 / 255.0,
                0.,
            )
            .expect("Conversion to f32 failed");

        // Resize mat to tensor width/height
        opencv::imgproc::resize(
            &self.f32_mat,
            &mut self.resized,
            opencv::core::Size::new(dimensions[3] as _, dimensions[2] as _),
            0.,
            0.,
            opencv::imgproc::INTER_LINEAR,
        )
        .expect("Resize failed");

        // Copy data
        let c = dimensions[1] as usize;
        let h = dimensions[2] as usize;
        let w = dimensions[3] as usize;

        let len = h * w * c;
        let ptr = self.resized.ptr(0).expect("Failed to get ptr") as *const f32;
        let src = unsafe { std::slice::from_raw_parts(ptr, len) };

        let hwc_view = ndarray::ArrayView3::from_shape((h, w, c), src)
            .expect("Shape mismatch between Mat and ndarray view");

        destination
            .extract_array_mut()
            .index_axis_mut(ndarray::Axis(0), 0)
            .assign(&hwc_view.permuted_axes([2, 0, 1]));
    }
}
