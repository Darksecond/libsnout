use std::time::Duration;

use snout::{
    calibration::{
        camera::{Calibration, FrameCalibrator},
        eye::EyeCalibrator,
        face::{FaceShape, ManualFaceCalibrator},
    },
    capture::{discovery::query_cameras, mono::MonoCamera, stereo::StereoCamera},
    output::{BabbleEmitter, OscTransport},
    pipeline::{eye::EyePipeline, face::FacePipeline, init_runtime},
};

pub fn main() {
    let sources = query_cameras();

    init_runtime();

    let mut camera = StereoCamera::open_sbs(sources[1].source).unwrap();
    let mut left_frame_calibrator = FrameCalibrator::new();
    let mut right_frame_calibrator = FrameCalibrator::new();
    let mut pipeline = EyePipeline::new("eyeModel.onnx").unwrap();
    let mut eye_calibrator = EyeCalibrator::new();

    loop {
        let (left, right) = camera.get_frames().unwrap();
        let left_calibrated_frame = left_frame_calibrator.calibrate(left).unwrap();
        let right_calibrated_frame = right_frame_calibrator.calibrate(right).unwrap();
        let Some(raw_weights) = pipeline
            .run(left_calibrated_frame, right_calibrated_frame)
            .unwrap()
        else {
            continue;
        };

        let weights = eye_calibrator.calibrate(raw_weights);

        println!("\x1B[2J\x1B[1;1H");
        dbg!(weights);

        std::thread::sleep(Duration::from_millis(10));
    }
}

pub fn main2() {
    let sources = query_cameras();

    // Run everything in a separate thread
    std::thread::spawn(move || {
        init_runtime();

        let mut camera = MonoCamera::open(sources[1].source).unwrap();
        let mut frame_calibrator = FrameCalibrator::new();

        frame_calibrator.set_calibration(Calibration {
            brightness: 1.,
            ..Calibration::default()
        });

        let mut pipeline = FacePipeline::new("faceModel.onnx").unwrap();
        let mut face_calibrator = ManualFaceCalibrator::new();
        let mut transport = OscTransport::udp("127.0.0.1:9400").unwrap();
        let mut babble = BabbleEmitter::new();

        loop {
            let frame = camera.get_frame().unwrap();
            let calibrated_frame = frame_calibrator.calibrate(frame).unwrap();
            let Some(weights) = pipeline.run(calibrated_frame).unwrap() else {
                continue;
            };

            let weights = face_calibrator.calibrate(weights);
            babble.process_face(weights, &mut transport);

            std::thread::sleep(Duration::from_millis(10));

            println!("\x1B[2J\x1B[1;1H");
            println!("JawOpen: {}", weights[FaceShape::JawOpen as usize].value);
            println!(
                "MouthSmileLeft: {}",
                weights[FaceShape::MouthSmileLeft as usize].value
            );
            println!(
                "MouthSmileRight: {}",
                weights[FaceShape::MouthSmileRight as usize].value
            );
        }
    })
    .join()
    .unwrap();
}
