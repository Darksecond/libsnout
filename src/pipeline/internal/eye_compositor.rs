use std::collections::VecDeque;

use image::GrayImage;
use imageproc::contrast::equalize_histogram;

use crate::capture::Frame;

/// In CHW layout
pub struct CompositeImage {
    pub data: Vec<u8>,

    pub width: u32,
    pub height: u32,

    pub channels: usize,
}

impl CompositeImage {
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            height: 0,
            channels: 8,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.data
            .resize((width * height * self.channels as u32) as usize, 0);
        self.width = width;
        self.height = height;
    }
}

pub struct EyeCompositor {
    composite: CompositeImage,
    queue: VecDeque<(GrayImage, GrayImage)>,
}

impl EyeCompositor {
    pub fn new() -> Self {
        Self {
            composite: CompositeImage::empty(),
            queue: VecDeque::with_capacity(5),
        }
    }

    pub fn compose(&mut self, left: &Frame, right: &Frame) -> Option<&CompositeImage> {
        let left_hist = equalize_histogram(&left.image);
        let right_hist = equalize_histogram(&right.image);

        self.queue.push_back((left_hist, right_hist));

        if self.queue.len() < 5 {
            return None;
        }

        self.queue.pop_front();

        // Unwrap is fine since we just tested that the queue is non-empty.
        let width = self.queue.back().unwrap().0.width();
        let height = self.queue.back().unwrap().0.height();

        self.composite.resize(width, height);

        let channels = self.queue.iter().rev().take(4).flat_map(|(l, r)| [r, l]);

        let pixels_per_channel = (width * height) as usize;

        for (c, ch) in channels.enumerate() {
            let start = c * pixels_per_channel;
            let end = start + pixels_per_channel;
            self.composite.data[start..end].copy_from_slice(ch.as_raw());
        }

        Some(&self.composite)
    }
}
