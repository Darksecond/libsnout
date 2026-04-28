mod error;

mod batcher;
mod data;
mod dataset;

use std::path::Path;
use std::sync::Arc;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::DataLoaderBuilder;
use burn::lr_scheduler::LrScheduler;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use image::{ImageBuffer, Luma};

use crate::cancel::Cancellation;
use crate::models::dual_eye_net::DualEyeNet;
use batcher::CaptureBatcher;
use data::DataReader;
use dataset::CaptureDataset;

pub use error::TrainerError;

pub(crate) type FloatImage = ImageBuffer<Luma<f32>, Vec<f32>>;

#[derive(Debug, Clone, Copy)]
pub struct Progress {
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub current_batch: u32,
    pub total_batches: u32,
    pub loss: f32,
}

#[derive(Debug, Clone)]
struct TrainingConfig {
    epochs_aug: u32,
    epochs_noaug: u32,
    batch_size: u32,
    learning_rate: f32,
    min_learning_rate: f32,
    num_workers: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs_aug: 8,
            epochs_noaug: 1,
            batch_size: 32,
            learning_rate: 1e-3,
            min_learning_rate: 1e-5,
            num_workers: 4,
        }
    }
}

pub struct Trainer {
    data: Arc<DataReader>,
    baseline: DualEyeNet<Autodiff<Wgpu>>,
    config: TrainingConfig,
    on_progress: Option<Box<dyn FnMut(Progress) + 'static>>,
}

impl Trainer {
    pub fn new(
        samples: impl AsRef<Path>,
        baseline: impl AsRef<Path>,
    ) -> Result<Self, TrainerError> {
        let device = WgpuDevice::default();

        let data = DataReader::from_file(samples.as_ref())
            .map_err(|source| TrainerError::Data(source.to_string()))?;

        let baseline = DualEyeNet::load_safetensors(baseline.as_ref(), &device)
            .map_err(|source| TrainerError::Baseline(source.to_string()))?;

        Ok(Self {
            data: Arc::new(data),
            baseline,
            config: TrainingConfig::default(),
            on_progress: None,
        })
    }

    pub fn on_progress<F>(&mut self, cb: F)
    where
        F: FnMut(Progress) + 'static,
    {
        self.on_progress = Some(Box::new(cb));
    }

    pub fn train(
        self,
        path: impl AsRef<Path>,
        cancel: &dyn Cancellation,
    ) -> Result<(), TrainerError> {
        let Trainer {
            data,
            baseline,
            config,
            mut on_progress,
        } = self;

        let batch_size = config.batch_size.max(1) as usize;
        let num_workers = config.num_workers.max(1) as usize;

        let dataloader_aug = DataLoaderBuilder::new(CaptureBatcher::new())
            .batch_size(batch_size)
            .num_workers(num_workers)
            .build(CaptureDataset::augmented(data.clone()));

        let dataloader_noaug = DataLoaderBuilder::new(CaptureBatcher::new())
            .batch_size(batch_size)
            .num_workers(num_workers)
            .build(CaptureDataset::plain(data));

        let aug_batches = Arc::clone(&dataloader_aug).num_items().div_ceil(batch_size);
        let noaug_batches = Arc::clone(&dataloader_noaug)
            .num_items()
            .div_ceil(batch_size);

        let total_iters = (config.epochs_aug as usize * aug_batches
            + config.epochs_noaug as usize * noaug_batches)
            .max(1);

        let total_epochs = config.epochs_aug + config.epochs_noaug;

        let mut lr_sched =
            CosineAnnealingLrSchedulerConfig::new(config.learning_rate as f64, total_iters)
                .with_min_lr(config.min_learning_rate as f64)
                .init()
                .expect("cosine scheduler params are in range");

        let mut model = baseline;
        let mut optim = AdamWConfig::new().init();

        for epoch in 1..=total_epochs {
            if cancel.is_cancelled() {
                return Err(TrainerError::Cancelled);
            }

            let augmented = epoch <= config.epochs_aug;
            let (dataloader, batches_per_epoch) = if augmented {
                (&dataloader_aug, aug_batches)
            } else {
                (&dataloader_noaug, noaug_batches)
            };
            let total_batches = batches_per_epoch as u32;
            let mut loss_sum: f32 = 0.0;

            for (batch_index, batch) in dataloader.iter().enumerate() {
                if cancel.is_cancelled() {
                    return Err(TrainerError::Cancelled);
                }

                let output = model.forward(batch.images);
                let loss = MseLoss::new().forward(output, batch.labels, Reduction::Mean);
                let loss_scalar = loss.clone().into_scalar().elem::<f32>();

                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                let lr = lr_sched.step();
                model = optim.step(lr, model, grads);

                loss_sum += loss_scalar;
                let batches_done = batch_index as u32 + 1;
                let mean_loss = loss_sum / batches_done as f32;

                if let Some(cb) = on_progress.as_mut() {
                    cb(Progress {
                        current_epoch: epoch,
                        total_epochs,
                        current_batch: batches_done,
                        total_batches,
                        loss: mean_loss,
                    });
                }
            }
        }

        model
            .valid()
            .save_onnx(path)
            .map_err(|_| TrainerError::Save("Failed to save onnx".to_string()))?;

        Ok(())
    }
}
