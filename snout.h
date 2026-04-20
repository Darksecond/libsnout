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
  SnoutError_Ok,
  SnoutError_CameraOpen,
  SnoutError_CameraInvalidFrame,
  SnoutError_CameraInternal,
} SnoutError;

typedef struct CameraSource CameraSource;

typedef struct Frame Frame;

typedef struct MonoCamera MonoCamera;

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

#endif  /* snout_h */
