use std::time::Duration;

use snout::{
    calibration::{
        camera::{Calibration, FrameCalibrator},
        face::{FaceShape, ManualFaceCalibrator},
    },
    capture::{discovery::query_cameras, mono::MonoCamera},
    output::{BabbleEmitter, OscTransport},
    pipeline::{face::FacePipeline, init_runtime},
};

pub fn main() {
    let sources = query_cameras();

    // Run everything in a separate thread
    std::thread::spawn(move || {
        init_runtime();

        let mut camera = MonoCamera::open(sources[0].source).unwrap();
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
