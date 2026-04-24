use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d, Relu, Sigmoid};
use burn::prelude::*;

pub(crate) const IMAGE_HEIGHT: usize = 128;
pub(crate) const IMAGE_WIDTH: usize = 128;
pub(crate) const PER_EYE_CHANNELS: usize = 4;
pub(crate) const PER_EYE_OUTPUTS: usize = 3;

const CONV_WIDTHS: [usize; 6] = [28, 42, 63, 94, 141, 212];
const EMBEDDING_DIMS: usize = CONV_WIDTHS[5];

#[derive(Module, Debug)]
pub struct EyeNet<B: Backend> {
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

impl<B: Backend> EyeNet<B> {
    pub fn new(device: &B::Device) -> EyeNet<B> {
        let conv = |in_c: usize, out_c: usize| -> Conv2dConfig {
            Conv2dConfig::new([in_c, out_c], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        };

        EyeNet {
            conv1: conv(PER_EYE_CHANNELS, CONV_WIDTHS[0]).init(device),
            conv2: conv(CONV_WIDTHS[0], CONV_WIDTHS[1]).init(device),
            conv3: conv(CONV_WIDTHS[1], CONV_WIDTHS[2]).init(device),
            conv4: conv(CONV_WIDTHS[2], CONV_WIDTHS[3]).init(device),
            conv5: conv(CONV_WIDTHS[3], CONV_WIDTHS[4]).init(device),
            conv6: conv(CONV_WIDTHS[4], CONV_WIDTHS[5]).init(device),
            fc: LinearConfig::new(EMBEDDING_DIMS, PER_EYE_OUTPUTS).init(device),
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
