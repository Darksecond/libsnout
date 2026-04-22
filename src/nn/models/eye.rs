use burn_store::{ApplyResult, SafetensorsStoreError};

mod micro;
mod multi;

pub use micro::MicroChad;
pub use multi::MultiChad;

fn validate(result: ApplyResult) -> Result<(), SafetensorsStoreError> {
    if !result.errors.is_empty() {
        return Err(SafetensorsStoreError::Other(format!(
            "safetensors apply reported errors: {:?}",
            result.errors
        )));
    }

    if !result.missing.is_empty() {
        return Err(SafetensorsStoreError::Other(format!(
            "safetensors file is missing expected tensors: {:?}",
            result.missing
        )));
    }

    Ok(())
}
