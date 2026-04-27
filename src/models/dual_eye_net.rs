use std::fs::File;
use std::io::Write;
use std::path::Path;

use burn::prelude::*;
use burn::store::{
    ApplyResult, BurnToPyTorchAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    SafetensorsStoreError,
};
use burn::tensor::Int;
use onnx_rs::ast::TensorProto;
use safetensors::SafeTensors;

use super::eye_net::{EyeNet, PER_EYE_CHANNELS, PER_EYE_OUTPUTS};

pub(crate) const HISTORY_LEN: usize = PER_EYE_CHANNELS;
pub(crate) const HISTORY_BASE: usize = HISTORY_LEN - 1;
pub(crate) const INPUT_CHANNELS: usize = 2 * PER_EYE_CHANNELS;
pub(crate) const LABEL_DIMS: usize = 2 * PER_EYE_OUTPUTS;

pub(crate) const LEFT_CHANNELS: [i64; PER_EYE_CHANNELS] = [0, 2, 4, 6];
pub(crate) const RIGHT_CHANNELS: [i64; PER_EYE_CHANNELS] = [1, 3, 5, 7];

#[derive(Module, Debug)]
pub struct DualEyeNet<B: Backend> {
    pub left: EyeNet<B>,
    pub right: EyeNet<B>,
}

impl<B: Backend> DualEyeNet<B> {
    /// Forward pass: `[B, 8, H, W] → [B, 6]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let device = x.device();

        let left_idx = Tensor::<B, 1, Int>::from_ints(LEFT_CHANNELS, &device);
        let right_idx = Tensor::<B, 1, Int>::from_ints(RIGHT_CHANNELS, &device);

        let left_input = x.clone().select(1, left_idx);
        let right_input = x.select(1, right_idx);

        let left_out = self.left.forward(left_input);
        let right_out = self.right.forward(right_input);

        Tensor::cat(vec![left_out, right_out], 1)
    }

    pub(crate) fn load_safetensors(
        path: impl AsRef<Path>,
        device: &B::Device,
    ) -> Result<Self, SafetensorsStoreError> {
        let mut store =
            SafetensorsStore::from_file(path.as_ref()).with_from_adapter(PyTorchToBurnAdapter);

        let mut model = DualEyeNet {
            left: EyeNet::new(device),
            right: EyeNet::new(device),
        };

        validate(model.load_from(&mut store)?)?;

        Ok(model)
    }

    pub(crate) fn save_safetensors(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(), SafetensorsStoreError> {
        let mut store =
            SafetensorsStore::from_file(path.as_ref()).with_to_adapter(BurnToPyTorchAdapter);

        self.save_into(&mut store)
    }

    // TODO: Make less bad
    pub(crate) fn save_onnx(&self, path: impl AsRef<Path>) -> Result<(), ()> {
        let mut store = SafetensorsStore::from_bytes(None).with_to_adapter(BurnToPyTorchAdapter);
        self.save_into(&mut store).unwrap();

        let bytes = store.get_bytes().unwrap();
        let safetensors = SafeTensors::deserialize(&bytes).unwrap();

        let bytes = std::fs::read("eyeModel.onnx").unwrap();
        let mut model = onnx_rs::parse(&bytes).unwrap();

        if let Some(graph) = &mut model.graph.as_mut() {
            for initializer in graph.initializer.iter_mut() {
                let name = initializer.name();

                if let Ok(view) = safetensors.tensor(name) {
                    println!("Updating weights for {}", name);

                    *initializer = TensorProto::from_raw(
                        initializer.name(),
                        initializer.dims().to_vec(),
                        initializer.data_type(),
                        view.data(),
                    );
                }
            }
        }

        let output_onnx = onnx_rs::encode(&model);
        let mut file = File::create(path).unwrap();
        file.write_all(&output_onnx).unwrap();

        Ok(())
    }
}

fn validate(result: ApplyResult) -> Result<(), SafetensorsStoreError> {
    if !result.errors.is_empty() {
        return Err(SafetensorsStoreError::Other(format!(
            "safetensors apply reported errors: {:?}",
            result.errors
        )));
    }

    if !result.missing.is_empty() {
        return Err(SafetensorsStoreError::Other(format!(
            "safetensors file is missing expected tensors: {:?}",
            result.missing
        )));
    }

    Ok(())
}
