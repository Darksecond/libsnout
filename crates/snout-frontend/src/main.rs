use snout::{
    calibration::FrameCalibrator,
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
        let mut calibrator = FrameCalibrator::new();
        let mut pipeline = FacePipeline::new("faceModel.onnx").unwrap();
        let mut transport = OscTransport::udp("127.0.0.1:9400").unwrap();
        let mut babble = BabbleEmitter::new();

        loop {
            let frame = camera.get_frame().unwrap();
            let calibrated_frame = calibrator.calibrate(frame).unwrap();
            let weights = pipeline.run(calibrated_frame).unwrap();

            if let Some(weights) = weights {
                babble.process_face(weights, &mut transport);
            }
        }
    })
    .join()
    .unwrap();
}
