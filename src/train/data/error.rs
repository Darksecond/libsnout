#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("failed to open capture: {0}")]
    Open(String),
    #[error("failed to read capture: {0}")]
    Read(String),
    #[error("invalid label: {0}")]
    InvalidLabel(String),
    #[error("no usable frames")]
    NoUsableFrames,
}
