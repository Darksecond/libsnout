use std::time::Duration;

use snout::{
    calibration::{
        camera::{Calibration, FrameCalibrator},
        eye::EyeCalibrator,
        face::ManualFaceCalibrator,
    },
    capture::{
        discovery::{CameraSource, query_cameras},
        mono::MonoCamera,
        stereo::StereoCamera,
    },
    output::{BabbleEmitter, EtvrEmitter, OscTransport},
    pipeline::{eye::EyePipeline, face::FacePipeline, init_runtime},
};

pub struct EyeTracker {
    camera: StereoCamera,
    left_frame_calibrator: FrameCalibrator,
    right_frame_calibrator: FrameCalibrator,
    pipeline: EyePipeline,
    eye_calibrator: EyeCalibrator,
    etvr: EtvrEmitter,
}

impl EyeTracker {
    pub fn new(source: CameraSource) -> Self {
        let camera = StereoCamera::open_sbs(source).unwrap();
        let left_frame_calibrator = FrameCalibrator::new();
        let right_frame_calibrator = FrameCalibrator::new();
        let pipeline = EyePipeline::new("eyeModel.onnx").unwrap();
        let eye_calibrator = EyeCalibrator::new();
        let etvr = EtvrEmitter::new();

        Self {
            camera,
            left_frame_calibrator,
            right_frame_calibrator,
            pipeline,
            eye_calibrator,
            etvr,
        }
    }

    pub fn track(&mut self, transport: &mut OscTransport) {
        let (left, right) = self.camera.get_frames().unwrap();
        let left_calibrated_frame = self.left_frame_calibrator.calibrate(left).unwrap();
        let right_calibrated_frame = self.right_frame_calibrator.calibrate(right).unwrap();
        let Some(raw_weights) = self
            .pipeline
            .run(left_calibrated_frame, right_calibrated_frame)
            .unwrap()
        else {
            return;
        };

        let weights = self.eye_calibrator.calibrate(raw_weights);

        self.etvr.process_eyes(weights, transport);
    }
}

struct FaceTracker {
    camera: MonoCamera,
    frame_calibrator: FrameCalibrator,
    pipeline: FacePipeline,
    face_calibrator: ManualFaceCalibrator,
    babble: BabbleEmitter,
}

impl FaceTracker {
    pub fn new(source: CameraSource) -> Self {
        let camera = MonoCamera::open(source).unwrap();
        let mut frame_calibrator = FrameCalibrator::new();
        let pipeline = FacePipeline::new("faceModel.onnx").unwrap();
        let face_calibrator = ManualFaceCalibrator::new();
        let babble = BabbleEmitter::new();

        frame_calibrator.set_calibration(Calibration {
            brightness: 1.,
            ..Calibration::default()
        });

        Self {
            camera,
            frame_calibrator,
            pipeline,
            face_calibrator,
            babble,
        }
    }

    pub fn track(&mut self, transport: &mut OscTransport) {
        let frame = self.camera.get_frame().unwrap();
        let calibrated_frame = self.frame_calibrator.calibrate(frame).unwrap();
        let Some(weights) = self.pipeline.run(calibrated_frame).unwrap() else {
            return;
        };

        let weights = self.face_calibrator.calibrate(weights);
        self.babble.process_face(weights, transport);
    }
}

pub fn main() {
    init_runtime();

    let sources = query_cameras();

    let mut transport = OscTransport::udp("127.0.0.1:9400").unwrap();
    let mut eye = EyeTracker::new(sources[1].source);
    let mut face = FaceTracker::new(sources[2].source);

    loop {
        eye.track(&mut transport);
        face.track(&mut transport);

        std::thread::sleep(Duration::from_millis(10));
    }
}
