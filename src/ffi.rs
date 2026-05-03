use std::sync::Mutex;
use std::{cell::RefCell, os::raw::c_char};

use crate::calibration::{
    Bounds, EyeCalibrator, EyeShape, FaceShape, ManualFaceCalibrator, Weights,
};
use crate::capture::processing::Crop;
use crate::capture::{
    CameraError, MonoCamera,
    discovery::{self, CameraInfo, CameraSource},
    processing::{FramePreprocessor, PreprocessConfig, PreprocessError},
};
use crate::capture::{Frame, StereoCamera};
use crate::output::{BabbleEmitter, EtvrEmitter, OscTransport, TransportError};
use crate::pipeline::{EyePipeline, FacePipeline, FilterParameters, PipelineError};
use crate::track::TrackerError;
use crate::track::eye::EyeTracker;
use crate::track::face::FaceTracker;
use crate::track::output::Output;

// TODO: thread_local!
static CAMERA_INFO: Mutex<Vec<CameraInfo>> = Mutex::new(Vec::new());

/// Represents an error that occurred during a Snout operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum SnoutError {
    /// The operation completed successfully.
    Ok,
    /// An null pointer was passed to a function that requires a valid pointer.
    NullPointer,
    /// The input string was not valid UTF-8.
    InvalidUtf8,
    /// The camera failed to open due to an invalid format.
    CameraInvalidFormat,
    /// An invalid frame was received from the camera.
    ///
    /// This might mean the camera was disconnected, or could be a transient error.
    CameraInvalidFrame,
    /// An internal error occurred during camera operations.
    CameraInternal,
    /// An internal error occurred during preprocessing.
    PreprocessInternal,
    /// The pipeline failed to load.
    PipelineLoad,
    /// The pipeline failed during inference.
    PipelineInference,
    /// The tracker failed to load the model.
    TrackerModel,
    /// The tracker failed to open the camera.
    TrackerOpen,
    /// An internal error occurred during tracking.
    TrackerInternal,
    /// Failed to bind the transport socket.
    TransportBind,
    /// Failed to resolve the transport destination address.
    TransportResolve,
}

impl From<TransportError> for SnoutError {
    fn from(error: TransportError) -> Self {
        match error {
            TransportError::Bind => SnoutError::TransportBind,
            TransportError::Resolve => SnoutError::TransportResolve,
        }
    }
}

impl From<TrackerError> for SnoutError {
    fn from(error: TrackerError) -> Self {
        match error {
            TrackerError::Model(_) => SnoutError::TrackerModel,
            TrackerError::Open(_) => SnoutError::TrackerOpen,
            TrackerError::Internal(_) => SnoutError::TrackerInternal,
        }
    }
}

impl From<CameraError> for SnoutError {
    fn from(error: CameraError) -> Self {
        match error {
            CameraError::InvalidFormat(_) => SnoutError::CameraInvalidFormat,
            CameraError::InvalidFrame(_) => SnoutError::CameraInvalidFrame,
            CameraError::Internal(_) => SnoutError::CameraInternal,
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

impl From<PipelineError> for SnoutError {
    fn from(error: PipelineError) -> Self {
        match error {
            PipelineError::Load(_) => SnoutError::PipelineLoad,
            PipelineError::Inference(_) => SnoutError::PipelineInference,
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

fn set_utf8_error(e: std::str::Utf8Error) {
    LAST_ERROR.with_borrow_mut(|last_error| {
        last_error.code = SnoutError::InvalidUtf8;
        last_error.message = e.to_string();
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

/// Get the display name for the camera at `index`.
///
/// Copies the display name into the buffer, null-terminating it.
/// The length of the display name, not including the null terminator, is returned.
///
/// If buffer is null or max_len is 0 then the length of the display name is returned.
#[unsafe(no_mangle)]
pub extern "C" fn snout_camera_display_name(
    index: usize,
    buffer: *mut c_char,
    max_len: usize,
) -> usize {
    clear_last_error();

    let cameras = CAMERA_INFO.lock().expect("Failed to acquire lock");

    let Some(info) = cameras.get(index) else {
        return 0;
    };

    let display_name = info.display_name();

    if buffer.is_null() || max_len == 0 {
        return display_name.len();
    }

    let copy_len = std::cmp::min(display_name.len(), max_len - 1);

    unsafe {
        std::ptr::copy_nonoverlapping(display_name.as_ptr(), buffer as *mut u8, copy_len);
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

/// Get the current preprocessing crop.
///
/// returns a copy of the current crop.
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_preprocessor_crop(preprocessor: *const FramePreprocessor) -> Crop {
    clear_last_error();

    if preprocessor.is_null() {
        set_null_pointer_error();
        return Crop::default();
    }

    let preprocessor = unsafe { &*preprocessor };

    preprocessor.crop()
}

/// Set the preprocessing crop.
#[unsafe(no_mangle)]
pub extern "C" fn snout_frame_preprocessor_set_crop(
    preprocessor: *mut FramePreprocessor,
    crop: *const Crop,
) {
    clear_last_error();

    if preprocessor.is_null() {
        set_null_pointer_error();
        return;
    }

    if crop.is_null() {
        set_null_pointer_error();
        return;
    }

    let crop = unsafe { *crop };

    let preprocessor = unsafe { &mut *preprocessor };

    preprocessor.set_crop(crop);
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

/// The number of face shape weights returned by [`snout_face_pipeline_run`].
#[unsafe(no_mangle)]
pub static SNOUT_FACE_SHAPE_COUNT: usize = 45;

/// Create a new face pipeline.
///
/// Returns a pointer to the pipeline.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_pipeline_new() -> *mut FacePipeline {
    clear_last_error();

    let pipeline = FacePipeline::new();

    Box::into_raw(Box::new(pipeline))
}

/// Set the model for the face pipeline from the given path.
///
/// Returns true if the model was loaded successfully, false otherwise.
/// Check [`snout_last_error`] for details.
///
/// If path is null, the model will be unloaded.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_pipeline_set_model(
    pipeline: *mut FacePipeline,
    path: *const c_char,
) -> bool {
    clear_last_error();

    let path = if path.is_null() {
        None
    } else {
        let path = unsafe { std::ffi::CStr::from_ptr(path) };

        Some(match path.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_utf8_error(e);
                return false;
            }
        })
    };

    if pipeline.is_null() {
        set_null_pointer_error();
        return false;
    }

    let pipeline = unsafe { &mut *pipeline };

    match pipeline.set_model(path) {
        Ok(()) => true,
        Err(e) => {
            set_last_error(e);
            false
        }
    }
}

/// Get the current filter parameters of the face pipeline.
///
/// Returns a copy of the current filter parameters.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_pipeline_filter(pipeline: *const FacePipeline) -> FilterParameters {
    clear_last_error();

    if pipeline.is_null() {
        set_null_pointer_error();
        return FilterParameters::default();
    }

    let pipeline = unsafe { &*pipeline };

    pipeline.filter()
}

/// Set the filter parameters of the face pipeline.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_pipeline_set_filter(
    pipeline: *mut FacePipeline,
    parameters: FilterParameters,
) {
    clear_last_error();

    if pipeline.is_null() {
        set_null_pointer_error();
        return;
    }

    let pipeline = unsafe { &mut *pipeline };

    pipeline.set_filter(parameters);
}

/// Run the face pipeline on a frame.
///
/// Returns a pointer to [`SNOUT_FACE_SHAPE_COUNT`] floats.
/// The returned array can be indexed using the [`FaceShape`] enum variants cast to an integer.
///
/// A returned null either indicates an error, or that the pipeline was not ready yet.
/// Check [`snout_get_last_error`] to determine which.
/// It will be `SnoutError_Ok` if the pipeline was not ready yet.
///
/// The returned pointer is valid until the next call to [`snout_face_pipeline_run`]
/// or [`snout_face_pipeline_free`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_pipeline_run(
    pipeline: *mut FacePipeline,
    frame: *const Frame,
) -> *const f32 {
    clear_last_error();

    if pipeline.is_null() || frame.is_null() {
        set_null_pointer_error();
        return std::ptr::null();
    }

    let pipeline = unsafe { &mut *pipeline };
    let frame = unsafe { &*frame };

    match pipeline.run(frame) {
        Ok(Some(weights)) => weights.as_ptr(),
        Ok(None) => std::ptr::null(),
        Err(e) => {
            set_last_error(e);
            std::ptr::null()
        }
    }
}

/// Free the face pipeline.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_pipeline_free(pipeline: *mut FacePipeline) {
    clear_last_error();

    if pipeline.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(pipeline));
    }
}

/// The number of eye shape weights returned by [`snout_eye_pipeline_run`].
#[unsafe(no_mangle)]
pub static SNOUT_EYE_SHAPE_COUNT: usize = 6;

/// Create a new eye pipeline.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_pipeline_new() -> *mut EyePipeline {
    clear_last_error();

    let pipeline = EyePipeline::new();
    Box::into_raw(Box::new(pipeline))
}

/// Set the model for the eye pipeline from the given path.
///
/// Returns true if the model was loaded successfully, false otherwise.
/// Check [`snout_last_error`] for details.
///
/// If path is null, the model will be unloaded.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_pipeline_set_model(
    pipeline: *mut EyePipeline,
    path: *const c_char,
) -> bool {
    clear_last_error();

    let path = if path.is_null() {
        None
    } else {
        let path = unsafe { std::ffi::CStr::from_ptr(path) };

        Some(match path.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_utf8_error(e);
                return false;
            }
        })
    };

    if pipeline.is_null() {
        set_null_pointer_error();
        return false;
    }

    let pipeline = unsafe { &mut *pipeline };

    match pipeline.set_model(path) {
        Ok(()) => true,
        Err(e) => {
            set_last_error(e);
            false
        }
    }
}

/// Get the current filter parameters of the eye pipeline.
///
/// Returns a copy of the current filter parameters.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_pipeline_filter(pipeline: *const EyePipeline) -> FilterParameters {
    clear_last_error();

    if pipeline.is_null() {
        set_null_pointer_error();
        return FilterParameters::default();
    }

    let pipeline = unsafe { &*pipeline };

    pipeline.filter()
}

/// Set the filter parameters of the eye pipeline.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_pipeline_set_filter(
    pipeline: *mut EyePipeline,
    parameters: FilterParameters,
) {
    clear_last_error();

    if pipeline.is_null() {
        set_null_pointer_error();
        return;
    }

    let pipeline = unsafe { &mut *pipeline };

    pipeline.set_filter(parameters);
}

/// Run the eye pipeline on a pair of stereo frames.
///
/// Returns a pointer to [`SNOUT_EYE_SHAPE_COUNT`] floats.
/// The returned array can be indexed using the [`EyeShape`] enum variants cast to an integer.
///
/// A returned null either indicates an error, or that the pipeline was not ready yet.
/// Check [`snout_last_error`] to determine which.
/// It will be `SnoutError_Ok` if the pipeline was not ready yet.
///
/// The returned pointer is valid until the next call to [`snout_eye_pipeline_run`]
/// or [`snout_eye_pipeline_free`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_pipeline_run(
    pipeline: *mut EyePipeline,
    left: *const Frame,
    right: *const Frame,
) -> *const f32 {
    clear_last_error();

    if pipeline.is_null() || left.is_null() || right.is_null() {
        set_null_pointer_error();
        return std::ptr::null();
    }

    let pipeline = unsafe { &mut *pipeline };
    let left = unsafe { &*left };
    let right = unsafe { &*right };

    match pipeline.run(left, right) {
        Ok(Some(weights)) => weights.as_ptr(),
        Ok(None) => std::ptr::null(),
        Err(e) => {
            set_last_error(e);
            std::ptr::null()
        }
    }
}

/// Free the eye pipeline.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_pipeline_free(pipeline: *mut EyePipeline) {
    clear_last_error();

    if pipeline.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(pipeline));
    }
}

/// Create a new face calibrator.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_calibrator_new() -> *mut ManualFaceCalibrator {
    clear_last_error();

    Box::into_raw(Box::new(ManualFaceCalibrator::new()))
}

/// Get the calibration bounds for a face shape.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_calibrator_bounds(
    calibrator: *const ManualFaceCalibrator,
    shape: FaceShape,
) -> Bounds {
    clear_last_error();

    if calibrator.is_null() {
        set_null_pointer_error();
        return Bounds::new();
    }

    let calibrator = unsafe { &*calibrator };

    calibrator.bounds(shape)
}

/// Set the calibration bounds for a face shape.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_calibrator_set_bounds(
    calibrator: *mut ManualFaceCalibrator,
    shape: FaceShape,
    bounds: Bounds,
) {
    clear_last_error();

    if calibrator.is_null() {
        set_null_pointer_error();
        return;
    }

    let calibrator = unsafe { &mut *calibrator };

    calibrator.set_bounds(shape, bounds);
}

/// Calibrate raw face weights.
///
/// `weights` must point to [`SNOUT_FACE_SHAPE_COUNT`] floats.
///
/// Returns a pointer to [`SNOUT_FACE_SHAPE_COUNT`] floats, or null if an error occurred.
/// The returned slice can be indexed using the [`FaceShape`] enum variants cast to an integer.
///
/// The returned pointer is valid until the next call to [`snout_face_calibrator_calibrate`]
/// or [`snout_face_calibrator_free`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_calibrator_calibrate(
    calibrator: *mut ManualFaceCalibrator,
    weights: *const f32,
) -> *const f32 {
    clear_last_error();

    if calibrator.is_null() || weights.is_null() {
        set_null_pointer_error();
        return std::ptr::null();
    }

    let calibrator = unsafe { &mut *calibrator };
    let weights = unsafe { std::slice::from_raw_parts(weights, SNOUT_FACE_SHAPE_COUNT) };

    calibrator.calibrate(weights).as_ptr()
}

/// Free the face calibrator.
///
/// Does nothing if the pointer is null.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_calibrator_free(calibrator: *mut ManualFaceCalibrator) {
    clear_last_error();

    if calibrator.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(calibrator));
    }
}

/// Create a new eye calibrator.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_calibrator_new() -> *mut EyeCalibrator {
    clear_last_error();

    Box::into_raw(Box::new(EyeCalibrator::new()))
}

/// Get the calibration bounds for an eye shape.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_calibrator_bounds(
    calibrator: *const EyeCalibrator,
    shape: EyeShape,
) -> Bounds {
    clear_last_error();

    if calibrator.is_null() {
        set_null_pointer_error();
        return Bounds::new();
    }

    let calibrator = unsafe { &*calibrator };

    calibrator.bounds(shape)
}

/// Set the calibration bounds for an eye shape.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_calibrator_set_bounds(
    calibrator: *mut EyeCalibrator,
    shape: EyeShape,
    bounds: Bounds,
) {
    clear_last_error();

    if calibrator.is_null() {
        set_null_pointer_error();
        return;
    }

    let calibrator = unsafe { &mut *calibrator };

    calibrator.set_bounds(shape, bounds);
}

/// Get whether the eye calibrator links the eyes.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_calibrator_link_eyes(calibrator: *const EyeCalibrator) -> bool {
    clear_last_error();

    if calibrator.is_null() {
        set_null_pointer_error();
        return false;
    }

    let calibrator = unsafe { &*calibrator };

    calibrator.link_eyes()
}

/// Set whether the eye calibrator links the eyes.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_calibrator_set_link_eyes(
    calibrator: *mut EyeCalibrator,
    link_eyes: bool,
) {
    clear_last_error();

    if calibrator.is_null() {
        set_null_pointer_error();
        return;
    }

    let calibrator = unsafe { &mut *calibrator };

    calibrator.set_link_eyes(link_eyes);
}

/// Calibrate raw eye weights.
///
/// `weights` must point to [`SNOUT_EYE_SHAPE_COUNT`] floats.
///
/// Returns a pointer to [`SNOUT_EYE_SHAPE_COUNT`] floats, or null if an error occurred.
/// The returned slice can be indexed using the [`EyeShape`] enum.
///
/// The returned pointer is valid until the next call to [`snout_eye_calibrator_calibrate`]
/// or [`snout_eye_calibrator_free`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_calibrator_calibrate(
    calibrator: *mut EyeCalibrator,
    weights: *const f32,
) -> *const f32 {
    clear_last_error();

    if calibrator.is_null() || weights.is_null() {
        set_null_pointer_error();
        return std::ptr::null();
    }

    let calibrator = unsafe { &mut *calibrator };
    let weights = unsafe { std::slice::from_raw_parts(weights, SNOUT_EYE_SHAPE_COUNT) };

    calibrator.calibrate(weights).as_ptr()
}

/// Free the eye calibrator.
///
/// Does nothing if the pointer is null.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_calibrator_free(calibrator: *mut EyeCalibrator) {
    clear_last_error();

    if calibrator.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(calibrator));
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SnoutFaceReport {
    /// The raw frame.
    pub raw_frame: *const Frame,
    /// The frame after preprocessing.
    pub processed_frame: *const Frame,
    /// A pointer to [`SNOUT_FACE_SHAPE_COUNT`] floats.
    pub weights: *const f32,
}

impl SnoutFaceReport {
    const fn null() -> Self {
        Self {
            raw_frame: std::ptr::null(),
            processed_frame: std::ptr::null(),
            weights: std::ptr::null(),
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SnoutFaceTrackerFields {
    pub preprocessor: *mut FramePreprocessor,
    pub pipeline: *mut FacePipeline,
    pub calibrator: *mut ManualFaceCalibrator,
}

impl SnoutFaceTrackerFields {
    const fn null() -> Self {
        Self {
            preprocessor: std::ptr::null_mut(),
            pipeline: std::ptr::null_mut(),
            calibrator: std::ptr::null_mut(),
        }
    }
}

/// Creates a new [`FaceTracker`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_tracker_new() -> *mut FaceTracker {
    clear_last_error();

    let tracker = FaceTracker::new();

    Box::into_raw(Box::new(tracker))
}

/// Drops a [`FaceTracker`] instance created by [`snout_face_tracker_new`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_tracker_free(tracker: *mut FaceTracker) {
    clear_last_error();

    if tracker.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(tracker));
    }
}

/// Set the camera source for the [`FaceTracker`] instance.
///
/// If `source` is null, the camera will be closed.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_tracker_set_source(
    tracker: *mut FaceTracker,
    source: *const CameraSource,
) {
    clear_last_error();

    if tracker.is_null() {
        set_null_pointer_error();
        return;
    }

    let tracker = unsafe { &mut *tracker };

    let source = if source.is_null() {
        None
    } else {
        Some(unsafe { *source })
    };

    tracker.set_source(source);
}

/// Track a face using the [`FaceTracker`] instance.
///
/// Returns a null report if the tracker is null or an error occurs.
/// See [`snout_last_error`] for details.
///
/// If the error is [`SnoutError_Ok`], then there was insufficient data or a transient error.
/// Call [`snout_face_tracker_track`] again to retry.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_tracker_track(tracker: *mut FaceTracker) -> SnoutFaceReport {
    clear_last_error();

    if tracker.is_null() {
        set_null_pointer_error();
        return SnoutFaceReport::null();
    }

    let tracker = unsafe { &mut *tracker };

    match tracker.track() {
        Ok(Some(report)) => SnoutFaceReport {
            raw_frame: report.raw_frame,
            processed_frame: report.processed_frame,
            weights: report.weights.as_ptr(),
        },
        Ok(None) => SnoutFaceReport::null(),
        Err(e) => {
            set_last_error(e);
            SnoutFaceReport::null()
        }
    }
}

/// Returns the raw pointers to the [`FaceTracker`] fields.
///
/// This can be used for configuring the tracker.
/// Pointers are valid until [`snout_face_tracker_free`] is called.
#[unsafe(no_mangle)]
pub extern "C" fn snout_face_tracker_fields(tracker: *mut FaceTracker) -> SnoutFaceTrackerFields {
    clear_last_error();

    if tracker.is_null() {
        set_null_pointer_error();
        return SnoutFaceTrackerFields::null();
    }

    let tracker = unsafe { &mut *tracker };

    SnoutFaceTrackerFields {
        preprocessor: &mut tracker.preprocessor,
        pipeline: &mut tracker.pipeline,
        calibrator: &mut tracker.calibrator,
    }
}

/// Create a new UDP OSC transport.
///
/// `destination` is a null-terminated string like "127.0.0.1:9000".
/// Returns null if the socket could not be bound or the address could not be resolved.
/// See [`snout_last_error`] for details.
#[unsafe(no_mangle)]
pub extern "C" fn snout_osc_transport_udp(destination: *const c_char) -> *mut OscTransport {
    clear_last_error();

    if destination.is_null() {
        set_null_pointer_error();
        return std::ptr::null_mut();
    }

    let destination = unsafe { std::ffi::CStr::from_ptr(destination) };
    let destination = match destination.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_utf8_error(e);
            return std::ptr::null_mut();
        }
    };

    match OscTransport::udp(destination) {
        Ok(transport) => Box::into_raw(Box::new(transport)),
        Err(e) => {
            set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Free an OSC transport.
#[unsafe(no_mangle)]
pub extern "C" fn snout_osc_transport_free(transport: *mut OscTransport) {
    clear_last_error();

    if transport.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(transport));
    }
}

/// Flush the OSC transport.
///
/// Check [`snout_last_error`] to see if an error occurred.
#[unsafe(no_mangle)]
pub extern "C" fn snout_osc_transport_flush(transport: *mut OscTransport) {
    clear_last_error();

    if transport.is_null() {
        set_null_pointer_error();
        return;
    }

    let transport = unsafe { &mut *transport };

    if let Err(e) = transport.flush() {
        set_last_error(e);
    }
}

/// Create a new Babble emitter.
#[unsafe(no_mangle)]
pub extern "C" fn snout_babble_emitter_new() -> *mut BabbleEmitter {
    clear_last_error();

    Box::into_raw(Box::new(BabbleEmitter::new()))
}

/// Free a Babble emitter.
#[unsafe(no_mangle)]
pub extern "C" fn snout_babble_emitter_free(emitter: *mut BabbleEmitter) {
    clear_last_error();

    if emitter.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(emitter));
    }
}

/// Send face weights via the Babble protocol.
///
/// `weights` must point to [`SNOUT_FACE_SHAPE_COUNT`] floats.
#[unsafe(no_mangle)]
pub extern "C" fn snout_babble_emitter_process_face(
    emitter: *mut BabbleEmitter,
    weights: *const f32,
    transport: *mut OscTransport,
) {
    clear_last_error();

    if emitter.is_null() || weights.is_null() || transport.is_null() {
        set_null_pointer_error();
        return;
    }

    let emitter = unsafe { &mut *emitter };
    let weights = unsafe { std::slice::from_raw_parts(weights, SNOUT_FACE_SHAPE_COUNT) };
    let transport = unsafe { &mut *transport };

    emitter.process_face(Weights::new(weights), transport);
}

/// Create a new ETVR emitter.
#[unsafe(no_mangle)]
pub extern "C" fn snout_etvr_emitter_new() -> *mut EtvrEmitter {
    clear_last_error();

    Box::into_raw(Box::new(EtvrEmitter::new()))
}

/// Free an ETVR emitter.
#[unsafe(no_mangle)]
pub extern "C" fn snout_etvr_emitter_free(emitter: *mut EtvrEmitter) {
    clear_last_error();

    if emitter.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(emitter));
    }
}

/// Send eye weights via the ETVR protocol.
///
/// `weights` must point to [`SNOUT_EYE_SHAPE_COUNT`] floats.
#[unsafe(no_mangle)]
pub extern "C" fn snout_etvr_emitter_process_eyes(
    emitter: *mut EtvrEmitter,
    weights: *const f32,
    transport: *mut OscTransport,
) {
    clear_last_error();

    if emitter.is_null() || weights.is_null() || transport.is_null() {
        set_null_pointer_error();
        return;
    }

    let emitter = unsafe { &mut *emitter };
    let weights = unsafe { std::slice::from_raw_parts(weights, SNOUT_EYE_SHAPE_COUNT) };
    let transport = unsafe { &mut *transport };

    emitter.process_eyes(Weights::new(weights), transport);
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SnoutEyeReport {
    /// The raw left frame.
    pub left_raw_frame: *const Frame,
    /// The raw right frame.
    pub right_raw_frame: *const Frame,
    /// The left frame after preprocessing.
    pub left_processed_frame: *const Frame,
    /// The right frame after preprocessing.
    pub right_processed_frame: *const Frame,
    /// A pointer to [`SNOUT_EYE_SHAPE_COUNT`] floats, or null during warmup.
    pub weights: *const f32,
}

impl SnoutEyeReport {
    const fn null() -> Self {
        Self {
            left_raw_frame: std::ptr::null(),
            right_raw_frame: std::ptr::null(),
            left_processed_frame: std::ptr::null(),
            right_processed_frame: std::ptr::null(),
            weights: std::ptr::null(),
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SnoutEyeTrackerFields {
    pub left_preprocessor: *mut FramePreprocessor,
    pub right_preprocessor: *mut FramePreprocessor,
    pub pipeline: *mut EyePipeline,
    pub calibrator: *mut EyeCalibrator,
}

impl SnoutEyeTrackerFields {
    const fn null() -> Self {
        Self {
            left_preprocessor: std::ptr::null_mut(),
            right_preprocessor: std::ptr::null_mut(),
            pipeline: std::ptr::null_mut(),
            calibrator: std::ptr::null_mut(),
        }
    }
}

/// Creates a new [`EyeTracker`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_tracker_new() -> *mut EyeTracker {
    clear_last_error();

    let tracker = EyeTracker::new();
    Box::into_raw(Box::new(tracker))
}

/// Drops an [`EyeTracker`] instance created by [`snout_eye_tracker_new`].
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_tracker_free(tracker: *mut EyeTracker) {
    clear_last_error();

    if tracker.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(tracker));
    }
}

/// Set the camera sources for the [`EyeTracker`] instance.
///
/// If both sources are null, the camera will be closed.
/// If left and right point to the same source, the camera will be opened in side-by-side mode.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_tracker_set_source(
    tracker: *mut EyeTracker,
    left: *const CameraSource,
    right: *const CameraSource,
) {
    clear_last_error();

    if tracker.is_null() {
        set_null_pointer_error();
        return;
    }

    let tracker = unsafe { &mut *tracker };

    let left = if left.is_null() {
        None
    } else {
        Some(unsafe { *left })
    };

    let right = if right.is_null() {
        None
    } else {
        Some(unsafe { *right })
    };

    tracker.set_source(left, right);
}

/// Track eyes using the [`EyeTracker`] instance.
///
/// Returns a null report if the tracker is null or an error occurs.
/// See [`snout_last_error`] for details.
///
/// If the error is [`SnoutError_Ok`], then there was insufficient data or a transient error.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_tracker_track(tracker: *mut EyeTracker) -> SnoutEyeReport {
    clear_last_error();

    if tracker.is_null() {
        set_null_pointer_error();
        return SnoutEyeReport::null();
    }

    let tracker = unsafe { &mut *tracker };

    match tracker.track() {
        Ok(Some(report)) => SnoutEyeReport {
            left_raw_frame: report.left_raw_frame,
            right_raw_frame: report.right_raw_frame,
            left_processed_frame: report.left_processed_frame,
            right_processed_frame: report.right_processed_frame,
            weights: report.weights.as_ptr(),
        },
        Ok(None) => SnoutEyeReport::null(),
        Err(e) => {
            set_last_error(e);
            SnoutEyeReport::null()
        }
    }
}

/// Returns the raw pointers to the [`EyeTracker`] fields.
///
/// This can be used for configuring the tracker.
/// Pointers are valid until [`snout_eye_tracker_free`] is called.
#[unsafe(no_mangle)]
pub extern "C" fn snout_eye_tracker_fields(tracker: *mut EyeTracker) -> SnoutEyeTrackerFields {
    clear_last_error();

    if tracker.is_null() {
        set_null_pointer_error();
        return SnoutEyeTrackerFields::null();
    }

    let tracker = unsafe { &mut *tracker };

    SnoutEyeTrackerFields {
        left_preprocessor: &mut tracker.left_preprocessor,
        right_preprocessor: &mut tracker.right_preprocessor,
        pipeline: &mut tracker.pipeline,
        calibrator: &mut tracker.calibrator,
    }
}

// ── Output ──

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SnoutOutputFields {
    pub transport: *mut OscTransport,
    pub babble: *mut BabbleEmitter,
    pub etvr: *mut EtvrEmitter,
}

impl SnoutOutputFields {
    const fn null() -> Self {
        Self {
            transport: std::ptr::null_mut(),
            babble: std::ptr::null_mut(),
            etvr: std::ptr::null_mut(),
        }
    }
}

/// Create a new output with the given destination address.
///
/// `destination` is a null-terminated string like "127.0.0.1:9000".
/// Returns null if the socket could not be bound or the address could not be resolved.
/// See [`snout_last_error`] for details.
#[unsafe(no_mangle)]
pub extern "C" fn snout_output_new(destination: *const c_char) -> *mut Output {
    clear_last_error();

    if destination.is_null() {
        set_null_pointer_error();
        return std::ptr::null_mut();
    }

    let destination = unsafe { std::ffi::CStr::from_ptr(destination) };
    let destination = match destination.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_utf8_error(e);
            return std::ptr::null_mut();
        }
    };

    match Output::new(destination) {
        Ok(output) => Box::into_raw(Box::new(output)),
        Err(e) => {
            set_last_error(e);
            std::ptr::null_mut()
        }
    }
}

/// Free an output.
#[unsafe(no_mangle)]
pub extern "C" fn snout_output_free(output: *mut Output) {
    clear_last_error();

    if output.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(output));
    }
}

/// Set the destination address of the output.
///
/// `destination` is a null-terminated string like "127.0.0.1:9000".
#[unsafe(no_mangle)]
pub extern "C" fn snout_output_set_destination(output: *mut Output, destination: *const c_char) {
    clear_last_error();

    if output.is_null() || destination.is_null() {
        set_null_pointer_error();
        return;
    }

    let destination = unsafe { std::ffi::CStr::from_ptr(destination) };
    let destination = match destination.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_utf8_error(e);
            return;
        }
    };

    let output = unsafe { &mut *output };

    if let Err(e) = output.set_destination(destination) {
        set_last_error(e);
    }
}

/// Send face weights via all enabled face emitters.
///
/// `weights` must point to [`SNOUT_FACE_SHAPE_COUNT`] floats.
#[unsafe(no_mangle)]
pub extern "C" fn snout_output_send_face(output: *mut Output, weights: *const f32) {
    clear_last_error();

    if output.is_null() || weights.is_null() {
        set_null_pointer_error();
        return;
    }

    let output = unsafe { &mut *output };
    let weights = unsafe { std::slice::from_raw_parts(weights, SNOUT_FACE_SHAPE_COUNT) };

    output.send_face(Weights::new(weights));
}

/// Send eye weights via all enabled eye emitters.
///
/// `weights` must point to [`SNOUT_EYE_SHAPE_COUNT`] floats.
#[unsafe(no_mangle)]
pub extern "C" fn snout_output_send_eyes(output: *mut Output, weights: *const f32) {
    clear_last_error();

    if output.is_null() || weights.is_null() {
        set_null_pointer_error();
        return;
    }

    let output = unsafe { &mut *output };
    let weights = unsafe { std::slice::from_raw_parts(weights, SNOUT_EYE_SHAPE_COUNT) };

    output.send_eyes(Weights::new(weights));
}

/// Flush the output transport.
#[unsafe(no_mangle)]
pub extern "C" fn snout_output_flush(output: *mut Output) {
    clear_last_error();

    if output.is_null() {
        set_null_pointer_error();
        return;
    }

    let output = unsafe { &mut *output };

    if let Err(e) = output.flush() {
        set_last_error(e);
    }
}

/// Returns the raw pointers to the [`Output`] fields.
///
/// This can be used for direct access to the transport and emitters.
/// Pointers are valid until [`snout_output_free`] is called.
#[unsafe(no_mangle)]
pub extern "C" fn snout_output_fields(output: *mut Output) -> SnoutOutputFields {
    clear_last_error();

    if output.is_null() {
        set_null_pointer_error();
        return SnoutOutputFields::null();
    }

    let output = unsafe { &mut *output };

    SnoutOutputFields {
        transport: &mut output.transport,
        babble: &mut output.babble,
        etvr: &mut output.etvr,
    }
}
