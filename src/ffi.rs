use std::sync::Mutex;
use std::{cell::RefCell, os::raw::c_char};

use crate::capture::{
    CameraError, MonoCamera,
    discovery::{self, CameraInfo, CameraSource},
    processing::{FramePreprocessor, PreprocessConfig, PreprocessError},
};
use crate::capture::{Frame, StereoCamera};

// TODO: thread_local!
static CAMERA_INFO: Mutex<Vec<CameraInfo>> = Mutex::new(Vec::new());

/// Represents an error that occurred during a Snout operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum SnoutError {
    Ok,
    NullPointer,
    CameraOpen,
    CameraInvalidFrame,
    CameraInternal,
    CameraFrameMismatch,
    PreprocessInternal,
}

impl From<CameraError> for SnoutError {
    fn from(error: CameraError) -> Self {
        match error {
            CameraError::OpenError => SnoutError::CameraOpen,
            CameraError::InvalidFrame(_) => SnoutError::CameraInvalidFrame,
            CameraError::Internal(_) => SnoutError::CameraInternal,
            CameraError::FrameMismatch { .. } => SnoutError::CameraFrameMismatch,
        }
    }
}

impl From<PreprocessError> for SnoutError {
    fn from(error: PreprocessError) -> Self {
        match error {
            PreprocessError::Internal(_) => SnoutError::PreprocessInternal,
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

fn set_null_pointer_error() {
    LAST_ERROR.with_borrow_mut(|last_error| {
        last_error.code = SnoutError::NullPointer;
        last_error.message = "a required argument is null".to_string();
    });
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
    clear_last_error();

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
    clear_last_error();

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
    clear_last_error();

    let cameras = CAMERA_INFO.lock().expect("Failed to acquire lock");

    let Some(info) = cameras.get(index) else {
        return std::ptr::null_mut();
    };

    Box::into_raw(Box::new(info.source))
}

/// Free the camera source acquired by [`snout_camera_source`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_camera_source_free(source: *mut CameraSource) {
    clear_last_error();

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
    clear_last_error();

    if source.is_null() {
        set_null_pointer_error();
        return std::ptr::null_mut();
    }

    let source = unsafe { *source };

    match MonoCamera::open(source) {
        Ok(camera) => Box::into_raw(Box::new(camera)),
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
    clear_last_error();

    if camera.is_null() {
        set_null_pointer_error();
        return std::ptr::null();
    }

    let camera = unsafe { &mut *camera };

    match camera.get_frame() {
        Ok(frame) => frame as *const Frame,
        Err(e) => {
            set_last_error(e);
            std::ptr::null()
        }
    }
}

/// Free the mono camera acquired by [`snout_mono_camera_open`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_mono_camera_free(camera: *mut MonoCamera) {
    clear_last_error();

    if camera.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(camera as *mut MonoCamera));
    }
}

/// Get the width of the frame.
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_width(frame: *const Frame) -> usize {
    clear_last_error();

    if frame.is_null() {
        set_null_pointer_error();
        return 0;
    }

    let frame = unsafe { &*frame };

    frame.width()
}

/// Get the height of the frame.
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_height(frame: *const Frame) -> usize {
    clear_last_error();

    if frame.is_null() {
        set_null_pointer_error();
        return 0;
    }

    let frame = unsafe { &*frame };
    frame.height()
}

/// Get the data of the frame.
///
/// This will not take ownership of the data.
/// The data length is [`snout_frame_width`] * [`snout_frame_height`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_data(frame: *const Frame) -> *const u8 {
    clear_last_error();

    if frame.is_null() {
        set_null_pointer_error();
        return std::ptr::null();
    }

    let frame = unsafe { &*frame };

    frame.as_slice().as_ptr()
}

/// Open a stereo camera using the specified left and right camera sources.
///
/// Returns a pointer to the stereo camera, or null if the camera could not be opened.
/// Check [`snout_last_error`] for details.
#[unsafe(no_mangle)]
pub extern "C" fn snout_stereo_camera_open(
    left: *const CameraSource,
    right: *const CameraSource,
) -> *mut StereoCamera {
    clear_last_error();

    if left.is_null() || right.is_null() {
        set_null_pointer_error();
        return std::ptr::null_mut();
    }

    let left = unsafe { *left };
    let right = unsafe { *right };

    match StereoCamera::open(left, right) {
        Ok(camera) => Box::into_raw(Box::new(camera)),
        Err(e) => {
            set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Open a stereo camera using a single side-by-side source.
///
/// Returns a pointer to the stereo camera, or null if the camera could not be opened.
/// Check [`snout_last_error`] for details.
#[unsafe(no_mangle)]
pub extern "C" fn snout_stereo_camera_open_sbs(source: *const CameraSource) -> *mut StereoCamera {
    clear_last_error();

    if source.is_null() {
        set_null_pointer_error();

        return std::ptr::null_mut();
    }

    let source = unsafe { *source };

    match StereoCamera::open_sbs(source) {
        Ok(camera) => Box::into_raw(Box::new(camera)),
        Err(e) => {
            set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Free the stereo camera acquired by [`snout_stereo_camera_open`] or [`snout_stereo_camera_open_sbs`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_stereo_camera_free(camera: *mut StereoCamera) {
    clear_last_error();

    if camera.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(camera));
    }
}

/// Represents a pair of stereo camera frames.
#[repr(C)]
pub struct SnoutStereoCameraFrames {
    pub left: *const Frame,
    pub right: *const Frame,
}

/// Returns the stereo camera frames from the camera.
///
/// The returned [`SnoutStereoCameraFrames`] struct contains pointers to [`Frame`] instances.
/// The frames are valid until the [`snout_stereo_camera_free`] or [`snout_stereo_camera_get_frames`] function is called.
///
/// If an error occurs, the frames will be null and the error will be set.
#[unsafe(no_mangle)]
pub extern "C" fn snout_stereo_camera_get_frames(
    camera: *mut StereoCamera,
) -> SnoutStereoCameraFrames {
    clear_last_error();

    if camera.is_null() {
        set_null_pointer_error();
        return SnoutStereoCameraFrames {
            left: std::ptr::null(),
            right: std::ptr::null(),
        };
    }

    let camera = unsafe { &mut *camera };
    match camera.get_frames() {
        Ok((left, right)) => SnoutStereoCameraFrames {
            left: left as *const Frame,
            right: right as *const Frame,
        },
        Err(e) => {
            set_last_error(e);
            SnoutStereoCameraFrames {
                left: std::ptr::null(),
                right: std::ptr::null(),
            }
        }
    }
}

/// Create a new frame preprocessor.
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_preprocessor_new() -> *mut FramePreprocessor {
    clear_last_error();

    Box::into_raw(Box::new(FramePreprocessor::new()))
}

/// Free the frame preprocessor created by [`snout_frame_preprocessor_new`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_preprocessor_free(preprocessor: *mut FramePreprocessor) {
    clear_last_error();

    if preprocessor.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(preprocessor));
    }
}

/// Get the current preprocessing configuration.
///
/// returns a copy of the current configuration.
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_preprocessor_config(
    preprocessor: *const FramePreprocessor,
) -> PreprocessConfig {
    clear_last_error();

    if preprocessor.is_null() {
        set_null_pointer_error();
        return PreprocessConfig::default();
    }

    let preprocessor = unsafe { &*preprocessor };

    *preprocessor.config()
}

/// Set the preprocessing configuration.
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_preprocessor_set_config(
    preprocessor: *mut FramePreprocessor,
    config: PreprocessConfig,
) {
    clear_last_error();

    if preprocessor.is_null() {
        set_null_pointer_error();
        return;
    }

    let preprocessor = unsafe { &mut *preprocessor };

    preprocessor.set_config(config);
}

/// Process a frame using the preprocessor.
///
/// Returns a pointer to the processed frame, or null if an error occurred.
/// The returned frame is valid until the next call to [`snout_frame_preprocessor_process`]
/// or [`snout_frame_preprocessor_free`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_preprocessor_process(
    preprocessor: *mut FramePreprocessor,
    frame: *const Frame,
) -> *const Frame {
    clear_last_error();

    if preprocessor.is_null() || frame.is_null() {
        set_null_pointer_error();
        return std::ptr::null();
    }

    let preprocessor = unsafe { &mut *preprocessor };
    let frame = unsafe { &*frame };

    match preprocessor.process(frame) {
        Ok(result) => result as *const Frame,
        Err(e) => {
            set_last_error(e);
            std::ptr::null()
        }
    }
}
