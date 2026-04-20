use std::collections::VecDeque;

use opencv::{
    core::{Mat, Vector},
    imgproc,
};

use crate::capture::Frame;

pub struct EyeCollector {
    octo: Mat,
    queue: VecDeque<(Mat, Mat)>,
}

impl EyeCollector {
    pub fn new() -> Self {
        Self {
            octo: Mat::default(),
            queue: VecDeque::with_capacity(5),
        }
    }

    pub fn collect(&mut self, left: &Frame, right: &Frame) -> Result<Option<&Mat>, opencv::Error> {
        let mut left_hist = Mat::default();
        imgproc::equalize_hist(&left.mat, &mut left_hist)?;

        let mut right_hist = Mat::default();
        imgproc::equalize_hist(&right.mat, &mut right_hist)?;

        self.queue.push_back((left_hist, right_hist));

        if self.queue.len() < 5 {
            return Ok(None);
        }

        self.queue.pop_front();

        // TODO: Replace VecDequeue with `Vector::<Mat>` directly.
        let mut all_channels = Vector::<Mat>::with_capacity(8);

        for m in self.queue.iter().rev().take(4) {
            // Push right then left.
            all_channels.push(m.1.clone());
            all_channels.push(m.0.clone());
        }

        opencv::core::merge(&all_channels, &mut self.octo)?;

        Ok(Some(&self.octo))
    }
}
