use std::os::raw::c_char;
use std::sync::Mutex;

use crate::capture::discovery::{self, CameraInfo, CameraSource};

static CAMERA_INFO: Mutex<Vec<CameraInfo>> = Mutex::new(Vec::new());

/// Discover all available cameras.
///
/// Results are accessed via [`snout_camera_name`] and [`snout_camera_source`].
/// Returns the number of cameras found.
#[unsafe(no_mangle)]
pub extern "C" fn snout_query_cameras() -> usize {
    let mut cameras = CAMERA_INFO.lock().expect("Failed to acquire lock");

    *cameras = discovery::query_cameras();

    cameras.len()
}

/// Get the human-readable name for the camera at `index`.
///
/// Copies the name into the buffer, null-terminating it.
/// The length of the name, not including the null terminator, is returned.
///
/// If buffer is null or max_len is 0 then the length of the name is returned.
#[unsafe(no_mangle)]
pub extern "C" fn snout_camera_name(index: usize, buffer: *mut c_char, max_len: usize) -> usize {
    let cameras = CAMERA_INFO.lock().expect("Failed to acquire lock");

    let Some(info) = cameras.get(index) else {
        return 0;
    };

    if buffer.is_null() || max_len == 0 {
        return info.name.len();
    }

    let copy_len = std::cmp::min(info.name.len(), max_len - 1);

    unsafe {
        std::ptr::copy_nonoverlapping(info.name.as_ptr(), buffer as *mut u8, copy_len);
        *buffer.add(copy_len) = 0;
    }

    copy_len
}

/// Get the source for the camera at `index`.
///
/// Returns null if `index` is out of bounds.
/// The pointer is valid until [`snout_camera_source_free`] is called.
#[unsafe(no_mangle)]
pub extern "C" fn snout_camera_source(index: usize) -> *mut CameraSource {
    let cameras = CAMERA_INFO.lock().expect("Failed to acquire lock");

    let Some(info) = cameras.get(index) else {
        return std::ptr::null_mut();
    };

    Box::into_raw(Box::new(info.source))
}

/// Free the camera source acquired by [`snout_camera_source`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_camera_source_free(source: *mut CameraSource) {
    if source.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(source as *mut CameraSource));
    }
}
