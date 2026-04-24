use image::{GrayImage, ImageBuffer};
use imageproc::contrast::equalize_histogram;

use crate::models::dual_eye_net::LABEL_DIMS;
use crate::train::FloatImage;
use crate::train::data::DataError;
use crate::train::data::load::FrameMeta;

#[derive(Debug, Clone)]
pub struct CaptureFrame {
    /// `[L_pitch, L_yaw, L_lid, R_pitch, R_yaw, R_lid]`.
    pub label: [f32; LABEL_DIMS],
    /// Equalized + normalized left eye.
    pub left: FloatImage,
    /// Equalized + normalized right eye.
    pub right: FloatImage,
}

impl CaptureFrame {
    pub fn from_raw(
        meta: &FrameMeta,
        left_img: &GrayImage,
        right_img: &GrayImage,
    ) -> Result<Self, DataError> {
        Ok(CaptureFrame {
            label: compute_joint_label(meta)?,
            left: equalize_and_normalize(left_img),
            right: equalize_and_normalize(right_img),
        })
    }
}

fn compute_joint_label(m: &FrameMeta) -> Result<[f32; LABEL_DIMS], DataError> {
    let (lp, ly, ll) = per_eye_label(m.left_eye_pitch, m.left_eye_yaw, m.routine_left_lid, "left")?;
    let (rp, ry, rl) = per_eye_label(
        m.right_eye_pitch,
        m.right_eye_yaw,
        m.routine_right_lid,
        "right",
    )?;
    Ok([lp, ly, ll, rp, ry, rl])
}

fn per_eye_label(
    eye_pitch: f32,
    eye_yaw: f32,
    lid: f32,
    side: &'static str,
) -> Result<(f32, f32, f32), DataError> {
    let mut pitch = (eye_pitch + 45.0) / 90.0;
    let mut yaw = (eye_yaw + 45.0) / 90.0;

    if !(0.0..=1.0).contains(&pitch) || !(0.0..=1.0).contains(&yaw) {
        return Err(DataError::InvalidLabel(format!(
            "{side}: \
             eye_pitch={eye_pitch} eye_yaw={eye_yaw}"
        )));
    }

    let lid_flag = if lid < 0.5 {
        // Closed-eye override: pin gaze to neutral center and flip
        // lid_flag to 1 so the model's regression head isn't
        // penalized for "missing" an unfindable eye.
        pitch = 0.5;
        yaw = 0.5;
        1.0
    } else {
        0.0
    };

    Ok((pitch, yaw, lid_flag))
}

fn equalize_and_normalize(img: &GrayImage) -> FloatImage {
    let eq = equalize_histogram(img);
    gray_image_to_float_image(&eq)
}

/// Convert a `u8` grayscale image to an `f32` image with pixel values in `[0.0, 1.0]`.
fn gray_image_to_float_image(src: &GrayImage) -> FloatImage {
    let (w, h) = src.dimensions();
    let mut buf = Vec::with_capacity((w * h) as usize);
    for &p in src.as_raw() {
        buf.push(p as f32 / 255.0);
    }
    ImageBuffer::from_raw(w, h, buf).expect("shape matches")
}
