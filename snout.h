#ifndef snout_h
#define snout_h

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#define HISTORY_LEN PER_EYE_CHANNELS

#define HISTORY_BASE (HISTORY_LEN - 1)

#define INPUT_CHANNELS (2 * PER_EYE_CHANNELS)

#define LABEL_DIMS (2 * PER_EYE_OUTPUTS)

#define IMAGE_HEIGHT 128

#define IMAGE_WIDTH 128

#define PER_EYE_CHANNELS 4

#define PER_EYE_OUTPUTS 3

#define DEFAULT_THRESHOLD 0.022669

#define DEFAULT_ADAPTATION_WINDOW 100

#define FRAME_META_SIZE 100

/**
 * Sanity cap on per-eye JPEG size.
 */
#define MAX_JPEG_SIZE ((10 * 1024) * 1024)

/**
 * Represents an error that occurred during a Snout operation.
 */
typedef enum SnoutError {
  SnoutError_Ok,
  SnoutError_NullPointer,
  SnoutError_CameraOpen,
  SnoutError_CameraInvalidFrame,
  SnoutError_CameraInternal,
  SnoutError_CameraFrameMismatch,
} SnoutError;

typedef struct CameraSource CameraSource;

typedef struct Frame Frame;

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

#endif  /* snout_h */
