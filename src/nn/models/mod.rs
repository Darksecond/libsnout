pub mod face {
    include!(concat!(
        env!("OUT_DIR"),
        "/faceModel/simplifiedFaceModel.rs"
    ));
}

pub mod eye;
