use std::{thread::sleep, time::Duration};

use snout::{
    capture::discovery::query_cameras,
    config::Config,
    track::{eye::EyeTracker, face::FaceTracker, initialize_runtime, output::Output},
};

pub struct TrackCommand {
    config: Config,
}

impl TrackCommand {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn run(&self) {
        initialize_runtime(self.config.libonnxruntime.as_ref());

        let cameras = query_cameras();

        let mut face_tracker = FaceTracker::with_config(&cameras, &self.config).unwrap();
        let mut eye_tracker = EyeTracker::with_config(&cameras, &self.config).unwrap();

        let mut output = Output::with_config(&self.config).unwrap();

        println!("Tracking...");

        loop {
            let face_report = face_tracker.track().unwrap();
            let eye_report = eye_tracker.track().unwrap();

            if let Some(face_report) = face_report {
                output.send_face(face_report.weights);
            }

            if let Some(eye_report) = eye_report {
                output.send_eyes(eye_report.weights);
            }

            output.flush().unwrap();

            sleep(Duration::from_millis(10));
        }
    }
}
