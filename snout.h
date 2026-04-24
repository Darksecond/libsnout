#ifndef snout_h
#define snout_h

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Represents an error that occurred during a Snout operation.
 */
typedef enum SnoutError {
  /**
   * The operation completed successfully.
   */
  SnoutError_Ok,
  /**
   * An null pointer was passed to a function that requires a valid pointer.
   */
  SnoutError_NullPointer,
  /**
   * The input string was not valid UTF-8.
   */
  SnoutError_InvalidUtf8,
  /**
   * The camera failed to open.
   */
  SnoutError_CameraOpen,
  /**
   * An invalid frame was received from the camera.
   *
   * This might mean the camera was disconnected, or could be a transient error.
   */
  SnoutError_CameraInvalidFrame,
  /**
   * An internal error occurred during camera operations.
   */
  SnoutError_CameraInternal,
  /**
   * The camera frame did not match the expected format.
   */
  SnoutError_CameraFrameMismatch,
  /**
   * An internal error occurred during preprocessing.
   */
  SnoutError_PreprocessInternal,
  /**
   * The pipeline failed to load.
   */
  SnoutError_PipelineLoad,
  /**
   * The pipeline failed during inference.
   */
  SnoutError_PipelineInference,
} SnoutError;

typedef struct CameraSource CameraSource;

typedef struct FacePipeline FacePipeline;

typedef struct Frame Frame;

typedef struct FramePreprocessor FramePreprocessor;

typedef struct MonoCamera MonoCamera;

typedef struct StereoCamera StereoCamera;

/**
 * Represents a pair of stereo camera frames.
 */
typedef struct SnoutStereoCameraFrames {
  const struct Frame *left;
  const struct Frame *right;
} SnoutStereoCameraFrames;

/**
 * Crop an area of the frame.
 * defined by normalized coordinates (0.0 - 1.0).
 */
typedef struct Crop {
  float top;
  float left;
  float bottom;
  float right;
} Crop;

typedef struct PreprocessConfig {
  /**
   * In radians
   */
  float rotation;
  float brightness;
  bool horizontal_flip;
  bool vertical_flip;
  struct Crop crop;
} PreprocessConfig;

typedef struct FilterParameters {
  bool enable;
  float min_cutoff;
  float beta;
} FilterParameters;

/**
 * The number of face shape weights returned by [`snout_face_pipeline_run`].
 */
extern const uintptr_t SNOUT_FACE_SHAPE_COUNT;

/**
 * Get the last error that occurred.
 *
 * Returns the last error code on this thread.
 */
enum SnoutError snout_last_error(void);

/**
 * Copies the error message from the last fallible call into `buffer`.
 *
 * The message is null-terminated.
 * Returns the length of the message not including the null terminator.
 *
 * If `buffer` is null or `max_len` is 0, returns the length of the message.
 *
 * This will return the error message for this thread.
 */
uintptr_t snout_last_error_message(char *buffer, uintptr_t max_len);

/**
 * Discover all available cameras.
 *
 * Results are accessed via [`snout_camera_name`] and [`snout_camera_source`].
 * Returns the number of cameras found.
 */
uintptr_t snout_query_cameras(void);

/**
 * Get the human-readable name for the camera at `index`.
 *
 * Copies the name into the buffer, null-terminating it.
 * The length of the name, not including the null terminator, is returned.
 *
 * If buffer is null or max_len is 0 then the length of the name is returned.
 */
uintptr_t snout_camera_name(uintptr_t index, char *buffer, uintptr_t max_len);

/**
 * Get the source for the camera at `index`.
 *
 * Returns null if `index` is out of bounds.
 * The pointer is valid until [`snout_camera_source_free`] is called.
 */
struct CameraSource *snout_camera_source(uintptr_t index);

/**
 * Free the camera source acquired by [`snout_camera_source`].
 */
void snout_camera_source_free(struct CameraSource *source);

/**
 * Open a mono camera using the given source.
 *
 * Returns null if the camera could not be opened.
 * Check [`snout_last_error`] for details.
 */
struct MonoCamera *snout_mono_camera_open(const struct CameraSource *source);

/**
 * Get the next frame from the mono camera.
 *
 * Returns null if the frame could not be retrieved.
 * Check [`snout_last_error`] for details.
 *
 * The returned pointer is valid until the next call to [`snout_mono_camera_get_frame`] or [`snout_mono_camera_free`].
 */
const struct Frame *snout_mono_camera_get_frame(struct MonoCamera *camera);

/**
 * Free the mono camera acquired by [`snout_mono_camera_open`].
 */
void snout_mono_camera_free(struct MonoCamera *camera);

/**
 * Get the width of the frame.
 */
uintptr_t snout_frame_width(const struct Frame *frame);

/**
 * Get the height of the frame.
 */
uintptr_t snout_frame_height(const struct Frame *frame);

/**
 * Get the data of the frame.
 *
 * This will not take ownership of the data.
 * The data length is [`snout_frame_width`] * [`snout_frame_height`].
 */
const uint8_t *snout_frame_data(const struct Frame *frame);

/**
 * Open a stereo camera using the specified left and right camera sources.
 *
 * Returns a pointer to the stereo camera, or null if the camera could not be opened.
 * Check [`snout_last_error`] for details.
 */
struct StereoCamera *snout_stereo_camera_open(const struct CameraSource *left,
                                              const struct CameraSource *right);

/**
 * Open a stereo camera using a single side-by-side source.
 *
 * Returns a pointer to the stereo camera, or null if the camera could not be opened.
 * Check [`snout_last_error`] for details.
 */
struct StereoCamera *snout_stereo_camera_open_sbs(const struct CameraSource *source);

/**
 * Free the stereo camera acquired by [`snout_stereo_camera_open`] or [`snout_stereo_camera_open_sbs`].
 */
void snout_stereo_camera_free(struct StereoCamera *camera);

/**
 * Returns the stereo camera frames from the camera.
 *
 * The returned [`SnoutStereoCameraFrames`] struct contains pointers to [`Frame`] instances.
 * The frames are valid until the [`snout_stereo_camera_free`] or [`snout_stereo_camera_get_frames`] function is called.
 *
 * If an error occurs, the frames will be null and the error will be set.
 */
struct SnoutStereoCameraFrames snout_stereo_camera_get_frames(struct StereoCamera *camera);

/**
 * Create a new frame preprocessor.
 */
struct FramePreprocessor *snout_frame_preprocessor_new(void);

/**
 * Free the frame preprocessor created by [`snout_frame_preprocessor_new`].
 */
void snout_frame_preprocessor_free(struct FramePreprocessor *preprocessor);

/**
 * Get the current preprocessing configuration.
 *
 * returns a copy of the current configuration.
 */
struct PreprocessConfig snout_frame_preprocessor_config(const struct FramePreprocessor *preprocessor);

/**
 * Set the preprocessing configuration.
 */
void snout_frame_preprocessor_set_config(struct FramePreprocessor *preprocessor,
                                         struct PreprocessConfig config);

/**
 * Process a frame using the preprocessor.
 *
 * Returns a pointer to the processed frame, or null if an error occurred.
 * The returned frame is valid until the next call to [`snout_frame_preprocessor_process`]
 * or [`snout_frame_preprocessor_free`].
 */
const struct Frame *snout_frame_preprocessor_process(struct FramePreprocessor *preprocessor,
                                                     const struct Frame *frame);

/**
 * Create a new face pipeline, loading the model from the given path.
 *
 * Returns a pointer to the pipeline, or null if the model could not be loaded.
 * Check [`snout_last_error`] for details.
 */
struct FacePipeline *snout_face_pipeline_new(const char *path);

/**
 * Get the current filter parameters of the face pipeline.
 *
 * Returns a copy of the current filter parameters.
 */
struct FilterParameters snout_face_pipeline_filter(const struct FacePipeline *pipeline);

/**
 * Set the filter parameters of the face pipeline.
 */
void snout_face_pipeline_set_filter(struct FacePipeline *pipeline,
                                    struct FilterParameters parameters);

/**
 * Run the face pipeline on a frame.
 *
 * Returns a pointer to [`SNOUT_FACE_SHAPE_COUNT`] floats.
 *
 * A returned null either indicates an error, or that the pipeline was not ready yet.
 * Check [`snout_get_last_error`] to determine which.
 * It will be `SnoutError_Ok` if the pipeline was not ready yet.
 *
 * The returned pointer is valid until the next call to [`snout_face_pipeline_run`]
 * or [`snout_face_pipeline_free`].
 */
const float *snout_face_pipeline_run(struct FacePipeline *pipeline, const struct Frame *frame);

/**
 * Free the face pipeline.
 */
void snout_face_pipeline_free(struct FacePipeline *pipeline);

#endif  /* snout_h */
