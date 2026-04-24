use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use snout::cancel::Cancel;
use snout::train::{Progress, Trainer};

struct Cli {
    capture_path: String,
    output_path: PathBuf,
}

fn baseline_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("baseline.safetensors")
}

fn print_usage() {
    eprintln!("usage: train <capture.bin> <output.safetensors>");
}

fn parse_cli<I: IntoIterator<Item = String>>(args: I) -> Result<Cli, u8> {
    let mut positionals: Vec<String> = Vec::new();

    for arg in args {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Err(0);
            }
            s if s.starts_with("--") => {
                eprintln!("unknown flag: {s}");
                print_usage();
                return Err(2);
            }
            _ => positionals.push(arg),
        }
    }

    let mut iter = positionals.into_iter();
    let capture_path = match iter.next() {
        Some(p) => p,
        None => {
            print_usage();
            return Err(2);
        }
    };
    let output_path = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            print_usage();
            return Err(2);
        }
    };
    if iter.next().is_some() {
        eprintln!("unexpected extra positional argument");
        print_usage();
        return Err(2);
    }

    Ok(Cli {
        capture_path,
        output_path,
    })
}

/// Per-batch progress printer. Within an epoch, overwrites the same
/// terminal line with `\r`; at the last batch of each epoch, finishes
/// with a newline so the next epoch starts on a fresh line. Writes to
/// stdout and flushes immediately so the line appears in real time
/// rather than getting buffered until the next newline.
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

fn try_main(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    println!("capture:   {}", cli.capture_path);
    println!("output:    {}", cli.output_path.display());

    println!("---");
    println!("training joint DualEyeNet...");
    let mut trainer = Trainer::new(&cli.capture_path, baseline_path())?;
    trainer.on_progress(print_progress);

    trainer.train(&cli.output_path, Cancel::never())?;
    println!("wrote:     {}", cli.output_path.display());

    println!("---");
    println!("training completed successfully.");

    Ok(())
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = match parse_cli(env::args().skip(1)) {
        Ok(c) => c,
        Err(code) => return ExitCode::from(code),
    };

    match try_main(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
