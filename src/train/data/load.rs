use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

pub const FRAME_META_SIZE: usize = 100;

/// Sanity cap on per-eye JPEG size.
pub const MAX_JPEG_SIZE: usize = 10 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMeta {
    pub routine_pitch: f32,
    pub routine_yaw: f32,
    pub routine_distance: f32,
    pub routine_convergence: f32,
    pub fov_adjust_distance: f32,
    pub left_eye_pitch: f32,
    pub left_eye_yaw: f32,
    pub right_eye_pitch: f32,
    pub right_eye_yaw: f32,
    pub routine_left_lid: f32,
    pub routine_right_lid: f32,
    pub routine_brow_raise: f32,
    pub routine_brow_angry: f32,
    pub routine_widen: f32,
    pub routine_squint: f32,
    pub routine_dilate: f32,

    pub timestamp: i64,
    pub video_timestamp_left: i64,
    pub video_timestamp_right: i64,

    pub routine_state: i32,
    pub jpeg_left_len: i32,
    pub jpeg_right_len: i32,
}

#[derive(Debug, Clone)]
pub struct RawFrame {
    pub meta: FrameMeta,
    pub jpeg_left: Vec<u8>,
    pub jpeg_right: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("invalid JPEG length: left={left}, right={right}")]
    InvalidJpegLength { left: i32, right: i32 },

    #[error("JPEG length exceeds {MAX_JPEG_SIZE}-byte guard: left={left}, right={right}")]
    JpegTooLarge { left: i32, right: i32 },
}

pub struct CaptureReader<R: Read> {
    inner: R,
}

impl CaptureReader<BufReader<File>> {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self::new(BufReader::new(file)))
    }
}

impl<R: Read> CaptureReader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    pub fn next_frame(&mut self) -> Result<Option<RawFrame>, ReadError> {
        let meta = match self.read_meta()? {
            Some(m) => m,
            None => return Ok(None),
        };

        if meta.jpeg_left_len < 0 || meta.jpeg_right_len < 0 {
            return Err(ReadError::InvalidJpegLength {
                left: meta.jpeg_left_len,
                right: meta.jpeg_right_len,
            });
        }

        if (meta.jpeg_left_len as usize) > MAX_JPEG_SIZE
            || (meta.jpeg_right_len as usize) > MAX_JPEG_SIZE
        {
            return Err(ReadError::JpegTooLarge {
                left: meta.jpeg_left_len,
                right: meta.jpeg_right_len,
            });
        }

        let mut jpeg_left = vec![0u8; meta.jpeg_left_len as usize];
        self.inner.read_exact(&mut jpeg_left)?;

        let mut jpeg_right = vec![0u8; meta.jpeg_right_len as usize];
        self.inner.read_exact(&mut jpeg_right)?;

        Ok(Some(RawFrame {
            meta,
            jpeg_left,
            jpeg_right,
        }))
    }

    fn read_meta(&mut self) -> Result<Option<FrameMeta>, ReadError> {
        let mut buf = [0u8; FRAME_META_SIZE];

        let mut first = [0u8; 1];
        match self.inner.read(&mut first)? {
            0 => return Ok(None),
            _ => buf[0] = first[0],
        }
        self.inner.read_exact(&mut buf[1..])?;

        let mut cur = &buf[..];

        let routine_pitch = cur.read_f32::<LittleEndian>()?;
        let routine_yaw = cur.read_f32::<LittleEndian>()?;
        let routine_distance = cur.read_f32::<LittleEndian>()?;
        let routine_convergence = cur.read_f32::<LittleEndian>()?;
        let fov_adjust_distance = cur.read_f32::<LittleEndian>()?;
        let left_eye_pitch = cur.read_f32::<LittleEndian>()?;
        let left_eye_yaw = cur.read_f32::<LittleEndian>()?;
        let right_eye_pitch = cur.read_f32::<LittleEndian>()?;
        let right_eye_yaw = cur.read_f32::<LittleEndian>()?;
        let routine_left_lid = cur.read_f32::<LittleEndian>()?;
        let routine_right_lid = cur.read_f32::<LittleEndian>()?;
        let routine_brow_raise = cur.read_f32::<LittleEndian>()?;
        let routine_brow_angry = cur.read_f32::<LittleEndian>()?;
        let routine_widen = cur.read_f32::<LittleEndian>()?;
        let routine_squint = cur.read_f32::<LittleEndian>()?;
        let routine_dilate = cur.read_f32::<LittleEndian>()?;

        let timestamp = cur.read_i64::<LittleEndian>()?;
        let video_timestamp_left = cur.read_i64::<LittleEndian>()?;
        let video_timestamp_right = cur.read_i64::<LittleEndian>()?;

        let routine_state = cur.read_i32::<LittleEndian>()?;
        let jpeg_left_len = cur.read_i32::<LittleEndian>()?;
        let jpeg_right_len = cur.read_i32::<LittleEndian>()?;

        debug_assert!(cur.is_empty(), "header parser did not consume all bytes");

        Ok(Some(FrameMeta {
            routine_pitch,
            routine_yaw,
            routine_distance,
            routine_convergence,
            fov_adjust_distance,
            left_eye_pitch,
            left_eye_yaw,
            right_eye_pitch,
            right_eye_yaw,
            routine_left_lid,
            routine_right_lid,
            routine_brow_raise,
            routine_brow_angry,
            routine_widen,
            routine_squint,
            routine_dilate,
            timestamp,
            video_timestamp_left,
            video_timestamp_right,
            routine_state,
            jpeg_left_len,
            jpeg_right_len,
        }))
    }
}

impl<R: Read> Iterator for CaptureReader<R> {
    type Item = Result<RawFrame, ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_frame().transpose()
    }
}
