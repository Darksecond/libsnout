mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::commands::{CaptureCommand, ListCamerasCommand, TrackCommand, TrainCommand};

fn main() {
    let cli = Args::parse();

    let config = snout::config::load(&cli.config).unwrap();

    match cli.command {
        Commands::ListCameras {} => ListCamerasCommand::new().run(),
        Commands::Train {
            source,
            destination,
        } => TrainCommand::new(source, destination, config.train.baseline).run(),
        Commands::Track {} => TrackCommand::new(config).run(),
        Commands::Capture {
            source,
            destination,
        } => CaptureCommand::new(config, source, destination).run(),
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(flatten_help = true)]
struct Args {
    #[arg(short, long, value_name = "config.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CaptureSource {
    LeftEye,
    RightEye,
    Face,
}

#[derive(Subcommand)]
enum Commands {
    /// List all available cameras.
    ListCameras {},
    /// Save a frame from the specified source.
    Capture {
        source: CaptureSource,
        #[arg(value_name = "output.jpeg")]
        destination: PathBuf,
    },
    /// Train the eye model based on the captured samples.
    Train {
        /// A file containing samples for training.
        #[arg(value_name = "user_cal.bin")]
        source: PathBuf,
        /// A destination `onnx` file.
        #[arg(value_name = "output.onnx")]
        destination: PathBuf,
    },
    /// Start tracking!
    Track {},
}
