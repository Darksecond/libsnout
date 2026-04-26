fn main() {
    let cameras = snout::capture::discovery::query_cameras();
    for camera in cameras {
        println!("{}", camera.name);
        println!("{}", camera.display_name());
        println!("{:?}", camera.source);
        println!("");
    }
}
