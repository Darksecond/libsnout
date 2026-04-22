use std::path::Path;

use burn::prelude::*;
use burn::tensor::Int;
use burn_store::{
    BurnToPyTorchAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    SafetensorsStoreError,
};

use super::micro::MicroChad;

pub const LEFT_CHANNELS: [i64; 4] = [0, 2, 4, 6];
pub const RIGHT_CHANNELS: [i64; 4] = [1, 3, 5, 7];

/// Dual-eye wrapper, holds two independent [`MicroChad`] heads.
#[derive(Module, Debug)]
pub struct MultiChad<B: Backend> {
    pub left: MicroChad<B>,
    pub right: MicroChad<B>,
}

impl<B: Backend> MultiChad<B> {
    pub fn new(device: &B::Device) -> MultiChad<B> {
        MultiChad {
            left: MicroChad::new(device),
            right: MicroChad::new(device),
        }
    }

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

    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), SafetensorsStoreError> {
        let mut store =
            SafetensorsStore::from_file(path.as_ref()).with_to_adapter(BurnToPyTorchAdapter);

        self.save_into(&mut store)
    }

    pub fn load_safetensors<P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> Result<Self, SafetensorsStoreError> {
        let mut store =
            SafetensorsStore::from_file(path.as_ref()).with_from_adapter(PyTorchToBurnAdapter);

        let mut model = MultiChad::new(device);

        super::validate(model.load_from(&mut store)?)?;

        Ok(model)
    }
}
