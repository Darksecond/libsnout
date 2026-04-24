mod capture_frame;
mod corruption;
mod error;
mod load;

use std::io;
use std::path::Path;

use image::{GrayImage, ImageFormat, load_from_memory_with_format};

use capture_frame::CaptureFrame;
use corruption::CorruptionDetector;
use load::{CaptureReader, ReadError};

use crate::models::dual_eye_net::{HISTORY_BASE, HISTORY_LEN};

use error::DataError;

#[derive(Debug, Clone)]
pub struct DataReader {
    frames: Vec<CaptureFrame>,
}

impl DataReader {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, DataError> {
        let path = path.as_ref();

        tracing::info!(path = %path.display(), "loading training data");

        let mut detector = CorruptionDetector::default();

        let reader = CaptureReader::open(path).map_err(|e| DataError::Open(e.to_string()))?;

        let mut frames = Vec::new();
        let mut raw_frames: u32 = 0;
        let mut decode_failures: u32 = 0;
        let mut corrupted_pairs: u32 = 0;
        let mut truncated = false;

        for (idx, result) in reader.enumerate() {
            let raw = match result {
                Ok(r) => r,
                Err(ReadError::Io(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    truncated = true;
                    tracing::warn!(frame = idx, "capture file truncated mid-record");
                    break;
                }
                Err(e) => {
                    return Err(DataError::Read(e.to_string()));
                }
            };

            raw_frames += 1;

            let left_img = match decode_jpeg(&raw.jpeg_left) {
                Ok(img) => img,
                Err(e) => {
                    decode_failures += 1;
                    tracing::warn!(frame = idx, side = "left", error = %e, "JPEG decode failed");
                    continue;
                }
            };
            let right_img = match decode_jpeg(&raw.jpeg_right) {
                Ok(img) => img,
                Err(e) => {
                    decode_failures += 1;
                    tracing::warn!(frame = idx, side = "right", error = %e, "JPEG decode failed");
                    continue;
                }
            };

            let verdict = detector.process_frame_pair(&left_img, &right_img);
            if verdict.any_corrupted() {
                corrupted_pairs += 1;
                continue;
            }

            frames.push(CaptureFrame::from_raw(&raw.meta, &left_img, &right_img)?);
        }

        let clean_frames = frames.len() as u32;

        let data = DataReader { frames };
        let usable_frames = data.usable_len();

        tracing::info!(
            raw_frames,
            decode_failures,
            corrupted_pairs,
            clean_frames,
            usable_frames,
            truncated,
            "training data loaded",
        );

        if usable_frames == 0 {
            return Err(DataError::NoUsableFrames);
        }

        Ok(data)
    }

    pub fn usable_len(&self) -> u32 {
        self.frames.len().saturating_sub(HISTORY_BASE) as u32
    }

    /// Returns `[t, t-1, t-2, t-3]`.
    pub fn history_window(&self, idx: u32) -> Option<[&CaptureFrame; HISTORY_LEN]> {
        let cur_idx = self.frame_index(idx)?;

        Some(std::array::from_fn(|i| &self.frames[cur_idx - i]))
    }

    fn frame_index(&self, idx: u32) -> Option<usize> {
        let real = HISTORY_BASE + idx as usize;
        if real < self.frames.len() {
            Some(real)
        } else {
            None
        }
    }
}

fn decode_jpeg(jpeg_data: &[u8]) -> Result<GrayImage, image::ImageError> {
    Ok(load_from_memory_with_format(jpeg_data, ImageFormat::Jpeg)?.into_luma8())
}
