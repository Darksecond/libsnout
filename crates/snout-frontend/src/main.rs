use std::time::Duration;

use snout::{
    calibration::{EyeCalibrator, ManualFaceCalibrator},
    capture::{
        MonoCamera, StereoCamera,
        discovery::{CameraSource, query_cameras},
        processing::{FramePreprocessor, PreprocessConfig},
    },
    output::{BabbleEmitter, EtvrEmitter, OscTransport},
    pipeline::{EyePipeline, FacePipeline, init_runtime},
};

pub struct EyeTracker {
    camera: StereoCamera,
    left_preprocessor: FramePreprocessor,
    right_preprocessor: FramePreprocessor,
    pipeline: EyePipeline,
    eye_calibrator: EyeCalibrator,
    etvr: EtvrEmitter,
}

impl EyeTracker {
    pub fn new(source: CameraSource) -> Self {
        let camera = StereoCamera::open_sbs(source).unwrap();
        let left_frame_calibrator = FramePreprocessor::new();
        let right_frame_calibrator = FramePreprocessor::new();
        let pipeline = EyePipeline::new("eyeModel.onnx").unwrap();
        let eye_calibrator = EyeCalibrator::new();
        let etvr = EtvrEmitter::new();

        Self {
            camera,
            left_preprocessor: left_frame_calibrator,
            right_preprocessor: right_frame_calibrator,
            pipeline,
            eye_calibrator,
            etvr,
        }
    }

    pub fn track(&mut self, transport: &mut OscTransport) {
        let Ok((left, right)) = self.camera.get_frames() else {
            return;
        };

        let left_processed_frame = self.left_preprocessor.process(left).unwrap();
        let right_processed_frame = self.right_preprocessor.process(right).unwrap();
        let Some(raw_weights) = self
            .pipeline
            .run(left_processed_frame, right_processed_frame)
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
    preprocessor: FramePreprocessor,
    pipeline: FacePipeline,
    face_calibrator: ManualFaceCalibrator,
    babble: BabbleEmitter,
}

impl FaceTracker {
    pub fn new(source: CameraSource) -> Self {
        let camera = MonoCamera::open(source).unwrap();
        let mut preprocessor = FramePreprocessor::new();
        let pipeline = FacePipeline::new("faceModel.onnx").unwrap();
        let face_calibrator = ManualFaceCalibrator::new();
        let babble = BabbleEmitter::new();

        preprocessor.set_config(PreprocessConfig {
            brightness: 1.,
            ..PreprocessConfig::default()
        });

        Self {
            camera,
            preprocessor,
            pipeline,
            face_calibrator,
            babble,
        }
    }

    pub fn track(&mut self, transport: &mut OscTransport) {
        let Ok(frame) = self.camera.get_frame() else {
            return;
        };
        let processed_frame = self.preprocessor.process(frame).unwrap();
        let Some(weights) = self.pipeline.run(processed_frame).unwrap() else {
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
