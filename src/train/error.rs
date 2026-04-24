#[derive(Debug, thiserror::Error)]
pub enum TrainerError {
    #[error("training cancelled")]
    Cancelled,

    #[error("failed to load training data: {0}")]
    Data(String),

    #[error("failed to load baseline: {0}")]
    Baseline(String),

    #[error("failed to save model: {0}")]
    Save(String),
}
