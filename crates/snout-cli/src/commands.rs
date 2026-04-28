mod track;

use std::{io::Write, path::PathBuf};

use snout::{
    cancel::Cancel,
    capture::discovery::query_cameras,
    config::Config,
    pipeline::initialize_runtime,
    track::{eye::EyeTracker, face::FaceTracker},
    train::Progress,
};

pub use track::TrackCommand;

use crate::CaptureSource;

pub struct ListCamerasCommand {}

impl ListCamerasCommand {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(&self) {
        let cameras = snout::capture::discovery::query_cameras();
        for camera in cameras {
            println!("{}", camera.display_name());
        }
    }
}

pub struct TrainCommand {
    source: PathBuf,
    destination: PathBuf,
    baseline: PathBuf,
}

impl TrainCommand {
    pub fn new(source: PathBuf, destination: PathBuf, baseline: PathBuf) -> Self {
        Self {
            source,
            destination,
            baseline,
        }
    }

    pub fn run(&self) {
        println!("Training eye model...");
        let mut trainer = snout::train::Trainer::new(&self.source, &self.baseline).unwrap();
        trainer.on_progress(print_progress);
        trainer.train(&self.destination, Cancel::never()).unwrap();
        println!("wrote: {}", self.destination.display());
        println!("training completed successfully.");
    }
}

fn print_progress(p: Progress) {
    let line = format!(
        "epoch {:>2}/{:<2}  batch {:>4}/{:<4}  loss {:.5}",
        p.current_epoch, p.total_epochs, p.current_batch, p.total_batches, p.loss,
    );
    if p.current_batch == p.total_batches {
        // End of epoch — clear the in-place line and print with newline.
        println!("\r{line}");
    } else {
        print!("\r{line}");
        let _ = std::io::stdout().flush();
    }
}

pub struct CaptureCommand {
    config: Config,
    source: CaptureSource,
    destination: PathBuf,
}

impl CaptureCommand {
    pub fn new(config: Config, source: CaptureSource, destination: PathBuf) -> Self {
        Self {
            config,
            source,
            destination,
        }
    }

    pub fn run(&self) {
        let cameras = query_cameras();

        if let Some(path) = &self.config.libonnxruntime {
            initialize_runtime(path);
        } else {
            initialize_runtime("/usr/lib64/libonnxruntime.so");
        }

        {
            match self.source {
                CaptureSource::LeftEye => {
                    let mut tracker = EyeTracker::with_config(&cameras, &self.config).unwrap();

                    let mut i = 0;
                    while i < 10 {
                        if let Some(report) = tracker.track().unwrap() {
                            let frame = report.left_processed_frame.clone();
                            frame.into_image().save(&self.destination).unwrap();

                            println!("Captured frame!");
                            return;
                        }
                        i += 1;
                    }
                    println!("Could not capture frame");
                }
                CaptureSource::RightEye => {
                    let mut tracker = EyeTracker::with_config(&cameras, &self.config).unwrap();

                    let mut i = 0;
                    while i < 10 {
                        if let Some(report) = tracker.track().unwrap() {
                            let frame = report.right_processed_frame.clone();
                            frame.into_image().save(&self.destination).unwrap();

                            println!("Captured frame!");
                            return;
                        }

                        i += 1;
                    }
                    println!("Could not capture frame");
                }
                CaptureSource::Face => {
                    let mut tracker = FaceTracker::with_config(&cameras, &self.config).unwrap();

                    let mut i = 0;
                    while i < 10 {
                        if let Some(report) = tracker.track().unwrap() {
                            let frame = report.processed_frame.clone();
                            frame.into_image().save(&self.destination).unwrap();

                            println!("Captured frame!");
                            return;
                        }

                        i += 1;
                    }
                    println!("Could not capture frame");
                }
            }
        }
    }
}
