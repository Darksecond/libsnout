mod commands;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::commands::{ListCamerasCommand, TrainCommand};

fn main() {
    let cli = Args::parse();

    match cli.command {
        Commands::ListCameras {} => ListCamerasCommand::new().run(),
        Commands::Train {
            source,
            destination,
            baseline,
        } => TrainCommand::new(source, destination, Some(baseline)).run(),
    }
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(flatten_help = true)]
struct Args {
    #[arg(short, long, value_name = "config.toml")]
    config: Option<PathBuf>,

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
    /// Train the eye model based on the captured samples.
    Train {
        /// A file containing samples for training.
        #[arg(value_name = "user_cal.bin")]
        source: PathBuf,
        /// A destination `onnx` file.
        #[arg(value_name = "output.onnx")]
        destination: PathBuf,

        /// Baseline `safetensors` file to base the model on.
        #[arg(short, long, value_name = "eyeModel.safetensors")]
        baseline: PathBuf,
    },
}
