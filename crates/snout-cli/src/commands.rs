use std::{io::Write, path::PathBuf};

use snout::{cancel::Cancel, train::Progress};

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
    baseline: Option<PathBuf>,
}

impl TrainCommand {
    pub fn new(source: PathBuf, destination: PathBuf, baseline: Option<PathBuf>) -> Self {
        Self {
            source,
            destination,
            baseline,
        }
    }

    pub fn run(&self) {
        let Some(baseline) = &self.baseline else {
            println!("No baseline specified.");
            return;
        };

        println!("Training eye model...");
        let mut trainer = snout::train::Trainer::new(&self.source, baseline).unwrap();
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
