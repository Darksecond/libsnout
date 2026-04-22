use std::path::Path;

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu, Sigmoid};
use burn::prelude::*;
use burn_store::{
    BurnToPyTorchAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    SafetensorsStoreError,
};

pub const INPUT_CHANNELS: usize = 4;

pub const CONV_WIDTHS: [usize; 6] = [28, 42, 63, 94, 141, 212];
pub const OUTPUT_DIMS: usize = 3;
pub const EMBEDDING_DIMS: usize = CONV_WIDTHS[5];

/// Single-eye CNN regressor.
#[derive(Module, Debug)]
pub struct MicroChad<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
    pub conv4: Conv2d<B>,
    pub conv5: Conv2d<B>,
    pub conv6: Conv2d<B>,
    pub fc: Linear<B>,
    pub pool: MaxPool2d,
    pub act: Relu,
    pub sigmoid: Sigmoid,
}

impl<B: Backend> MicroChad<B> {
    pub fn new(device: &B::Device) -> MicroChad<B> {
        let conv = |in_c: usize, out_c: usize| -> Conv2dConfig {
            Conv2dConfig::new([in_c, out_c], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        };

        MicroChad {
            conv1: conv(INPUT_CHANNELS, CONV_WIDTHS[0]).init(device),
            conv2: conv(CONV_WIDTHS[0], CONV_WIDTHS[1]).init(device),
            conv3: conv(CONV_WIDTHS[1], CONV_WIDTHS[2]).init(device),
            conv4: conv(CONV_WIDTHS[2], CONV_WIDTHS[3]).init(device),
            conv5: conv(CONV_WIDTHS[3], CONV_WIDTHS[4]).init(device),
            conv6: conv(CONV_WIDTHS[4], CONV_WIDTHS[5]).init(device),
            fc: LinearConfig::new(EMBEDDING_DIMS, OUTPUT_DIMS).init(device),
            pool: MaxPool2dConfig::new([2, 2]).init(),
            act: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }

    /// Full forward pass: `[B, 4, 128, 128] → [B, 3]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let embedding = self.forward_embedding(x);
        let logits = self.fc.forward(embedding);
        self.sigmoid.forward(logits)
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

        let mut model = MicroChad::new(device);

        super::validate(model.load_from(&mut store)?)?;

        Ok(model)
    }

    fn forward_embedding(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.pool.forward(self.act.forward(self.conv1.forward(x)));
        let x = self.pool.forward(self.act.forward(self.conv2.forward(x)));
        let x = self.pool.forward(self.act.forward(self.conv3.forward(x)));
        let x = self.pool.forward(self.act.forward(self.conv4.forward(x)));
        let x = self.pool.forward(self.act.forward(self.conv5.forward(x)));

        let x = self.act.forward(self.conv6.forward(x));

        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, c, h * w]);
        let x = x.max_dim(2);
        x.flatten(1, 2)
    }
}
