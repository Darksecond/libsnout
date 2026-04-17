#ifndef snout_h
#define snout_h

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct CameraSource CameraSource;

/**
 * Discover all available cameras.
 *
 * Results are accessed via [`snout_camera_name`] and [`snout_camera_source`].
 * Returns the number of cameras found.
 *
 * Returned pointers are valid until the next call to [`snout_query_cameras`].
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

#endif  /* snout_h */
