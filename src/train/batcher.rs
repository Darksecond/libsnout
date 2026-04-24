use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::models::dual_eye_net::{INPUT_CHANNELS, LABEL_DIMS};
use crate::models::eye_net::{IMAGE_HEIGHT, IMAGE_WIDTH};
use crate::train::dataset::CaptureItem;

#[derive(Debug, Clone)]
pub(crate) struct CaptureBatch<B: Backend> {
    /// `[batch, INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH]`
    pub images: Tensor<B, 4>,
    /// `[batch, L_pitch, L_yaw, L_lid, R_pitch, R_yaw, R_lid]`
    pub labels: Tensor<B, 2>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CaptureBatcher<B: Backend> {
    _phantom: core::marker::PhantomData<B>,
}

impl<B: Backend> CaptureBatcher<B> {
    pub const fn new() -> Self {
        Self {
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, CaptureItem, CaptureBatch<B>> for CaptureBatcher<B> {
    fn batch(&self, items: Vec<CaptureItem>, device: &B::Device) -> CaptureBatch<B> {
        assert!(!items.is_empty(), "called with no items");

        const PER_ITEM: usize = INPUT_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;

        let batch = items.len();
        let mut flat_images: Vec<f32> = Vec::with_capacity(batch * PER_ITEM);
        let mut flat_labels: Vec<f32> = Vec::with_capacity(batch * LABEL_DIMS);

        for item in items {
            debug_assert_eq!(
                item.image.len(),
                PER_ITEM,
                "batch item image buffer length must equal INPUT_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH",
            );

            flat_images.extend_from_slice(&item.image);
            flat_labels.extend_from_slice(&item.label);
        }

        let images = Tensor::<B, 4>::from_data(
            TensorData::new(
                flat_images,
                [batch, INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH],
            ),
            device,
        );
        let labels =
            Tensor::<B, 2>::from_data(TensorData::new(flat_labels, [batch, LABEL_DIMS]), device);

        CaptureBatch { images, labels }
    }
}
