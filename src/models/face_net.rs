mod inner {
    #![allow(dead_code)]

    include!(concat!(
        env!("OUT_DIR"),
        "/faceModel/simplifiedFaceModel.rs"
    ));
}

pub use inner::Model as FaceNet;
