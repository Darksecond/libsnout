use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CameraSource {
    Index(u8),
}

#[derive(Clone, Debug)]
pub struct CameraInfo {
    pub source: CameraSource,
    pub name: String,
}

/// Queries the system for available cameras.
///
/// Returns a list of [`CameraInfo`] structs.
///
/// This will only work on Linux.
pub fn query_cameras() -> Vec<CameraInfo> {
    fn extract_index(path: &Path) -> Option<usize> {
        path.file_name()?
            .to_str()?
            .strip_prefix("video")?
            .parse()
            .ok()
    }

    fn is_primary_stream(index: usize) -> bool {
        let path = format!("/sys/class/video4linux/video{}/index", index);
        if let Ok(content) = std::fs::read_to_string(path) {
            if let Ok(idx) = content.trim().parse::<u32>() {
                return idx == 0;
            }
        }
        false
    }

    fn get_device_name(index: usize) -> String {
        let name_path = format!("/sys/class/video4linux/video{}/name", index);
        fs::read_to_string(name_path)
            .unwrap_or_else(|_| "Unknown Device".into())
            .trim()
            .to_string()
    }

    let mut cameras = HashMap::new();
    let by_path_dir = "/dev/v4l/by-path/";

    if let Ok(entries) = fs::read_dir(by_path_dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if let Ok(real_path) = fs::canonicalize(&path) {
                if let Some(index) = extract_index(&real_path) {
                    if is_primary_stream(index) {
                        let name = get_device_name(index);

                        cameras.insert(
                            index,
                            CameraInfo {
                                name,
                                source: CameraSource::Index(index as u8),
                            },
                        );
                    }
                }
            }
        }
    }

    let mut cameras: Vec<CameraInfo> = cameras.into_values().collect();

    cameras.sort_by_key(|c| match c.source {
        CameraSource::Index(i) => i,
    });

    cameras
}
