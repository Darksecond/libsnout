use std::sync::Mutex;
use std::{cell::RefCell, os::raw::c_char};

use crate::capture::Frame;
use crate::capture::{
    CameraError, MonoCamera,
    discovery::{self, CameraInfo, CameraSource},
};

// TODO: thread_local!
static CAMERA_INFO: Mutex<Vec<CameraInfo>> = Mutex::new(Vec::new());

/// Represents an error that occurred during a Snout operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum SnoutError {
    Ok,
    CameraOpen,
    CameraInvalidFrame,
    CameraInternal,
    CameraFrameMismatch,
}

impl From<CameraError> for SnoutError {
    fn from(error: CameraError) -> Self {
        match error {
            CameraError::OpenError => SnoutError::CameraOpen,
            CameraError::InvalidFrame => SnoutError::CameraInvalidFrame,
            CameraError::Internal(_) => SnoutError::CameraInternal,
            CameraError::FrameMismatch { .. } => SnoutError::CameraFrameMismatch,
        }
    }
}

struct LastError {
    code: SnoutError,
    message: String,
}

thread_local! {
    static LAST_ERROR: RefCell<LastError> = RefCell::new(LastError { code: SnoutError::Ok, message: String::new() })
}

fn set_last_error(e: impl Into<SnoutError> + std::error::Error) {
    LAST_ERROR.with_borrow_mut(|last_error| {
        last_error.message = e.to_string();
        last_error.code = e.into();
    });
}

fn clear_last_error() {
    LAST_ERROR.with_borrow_mut(|last_error| {
        last_error.code = SnoutError::Ok;
        last_error.message.clear();
    });
}

/// Get the last error that occurred.
///
/// Returns the last error code on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn snout_last_error() -> SnoutError {
    LAST_ERROR.with_borrow(|e| e.code)
}

/// Copies the error message from the last fallible call into `buffer`.
///
/// The message is null-terminated.
/// Returns the length of the message not including the null terminator.
///
/// If `buffer` is null or `max_len` is 0, returns the length of the message.
///
/// This will return the error message for this thread.
#[unsafe(no_mangle)]
pub extern "C" fn snout_last_error_message(buffer: *mut c_char, max_len: usize) -> usize {
    LAST_ERROR.with_borrow(|last_error| {
        if buffer.is_null() || max_len == 0 {
            return last_error.message.len();
        }

        let copy_len = std::cmp::min(last_error.message.len(), max_len - 1);

        unsafe {
            std::ptr::copy_nonoverlapping(last_error.message.as_ptr(), buffer as *mut u8, copy_len);
            *buffer.add(copy_len) = 0;
        }

        copy_len
    })
}

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

/// Open a mono camera using the given source.
///
/// Returns null if the camera could not be opened.
/// Check [`snout_last_error`] for details.
#[unsafe(no_mangle)]
pub extern "C" fn snout_mono_camera_open(source: *const CameraSource) -> *mut MonoCamera {
    let source = unsafe { *source };

    match MonoCamera::open(source) {
        Ok(camera) => {
            clear_last_error();
            Box::into_raw(Box::new(camera))
        }
        Err(e) => {
            set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Get the next frame from the mono camera.
///
/// Returns null if the frame could not be retrieved.
/// Check [`snout_last_error`] for details.
///
/// The returned pointer is valid until the next call to [`snout_mono_camera_get_frame`] or [`snout_mono_camera_free`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_mono_camera_get_frame(camera: *mut MonoCamera) -> *const Frame {
    let camera = unsafe { &mut *camera };

    match camera.get_frame() {
        Ok(frame) => {
            clear_last_error();
            frame as *const Frame
        }
        Err(e) => {
            set_last_error(e);
            std::ptr::null()
        }
    }
}

/// Free the mono camera acquired by [`snout_mono_camera_open`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_mono_camera_free(camera: *mut MonoCamera) {
    if camera.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(camera as *mut MonoCamera));
    }
}
