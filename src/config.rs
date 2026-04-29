use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::capture::processing::{Crop, PreprocessConfig};

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("No file found.")]
    FileNotFound,
    #[error("Invalid format: {0}")]
    InvalidConfig(String),
}

pub fn load(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
    let str = std::fs::read_to_string(path).map_err(|_| ConfigError::FileNotFound)?;
    let config = toml::from_str(&str).map_err(|e| ConfigError::InvalidConfig(e.to_string()))?;

    Ok(config)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub libonnxruntime: Option<PathBuf>,

    pub eye: EyesConfig,
    pub face: FaceConfig,
    pub train: TrainConfig,

    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EyesConfig {
    pub link: Option<bool>,
    pub model: Option<PathBuf>,

    pub left: EyeConfig,
    pub right: EyeConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EyeConfig {
    pub camera: String,
    pub crop: Option<Crop>,
    pub transform: Option<PreprocessConfig>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FaceConfig {
    pub camera: String,
    pub model: Option<PathBuf>,
    pub crop: Option<Crop>,
    pub transform: Option<PreprocessConfig>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TrainConfig {
    pub baseline: PathBuf,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct OutputConfig {
    #[serde(default)]
    pub osc: OscConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OscConfig {
    pub destination: String,
}

impl Default for OscConfig {
    fn default() -> Self {
        Self {
            destination: "127.0.0.1:9400".to_string(),
        }
    }
}
