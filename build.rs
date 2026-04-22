use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("simplifiedFaceModel.onnx")
        .out_dir("faceModel/")
        .run_from_script();

    // copy $OUT_DIR/faceModel/simplifiedFaceModel.bpk to Cargo.toml directory.
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let destination = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    std::fs::copy(
        format!("{}/faceModel/simplifiedFaceModel.bpk", out_dir),
        format!("{}/simplifiedFaceModel.bpk", destination),
    )
    .expect("Failed to copy simplifiedFaceModel.bpk");
}
