use v4l::Device;
use v4l::capability::Flags;
use v4l::frameinterval::FrameIntervalEnum;
use v4l::framesize::FrameSizeEnum;
use v4l::video::Capture;

/// Pixel format category for the camera.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum PixelFormat {
    /// Single-channel greyscale.
    Grey,
    /// Raw uncompressed (YUYV or UYVY).
    Raw,
    /// Motion JPEG compressed.
    Mjpeg,
}

/// Camera format describing resolution, framerate, and pixel format.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct CameraFormat {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub pixel_format: PixelFormat,
}

/// V4L2-specific camera source.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct V4lSource {
    pub index: u8,
    pub format: CameraFormat,

    /// The actual V4L2 FourCC code (e.g. `b"YUYV"`, `b"MJPG"`).
    /// Stored internally so we can set the exact format when opening.
    pub(crate) fourcc: [u8; 4],
}

/// Identifies a camera device and how to open it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CameraSource {
    V4l(V4lSource),
}

/// Information about a discovered camera.
#[derive(Clone, Debug)]
pub struct CameraInfo {
    pub name: String,
    pub source: CameraSource,
}

impl CameraInfo {
    /// Returns a human-readable name including resolution and framerate.
    pub fn display_name(&self) -> String {
        match &self.source {
            CameraSource::V4l(v4l) => {
                let format = &v4l.format;
                format!(
                    "{} ({}x{} @ {}fps)",
                    self.name, format.width, format.height, format.fps
                )
            }
        }
    }
}

fn fourcc_to_pixel_format(fourcc: &[u8; 4]) -> Option<PixelFormat> {
    match fourcc {
        b"GREY" => Some(PixelFormat::Grey),
        b"YUYV" | b"UYVY" => Some(PixelFormat::Raw),
        b"MJPG" => Some(PixelFormat::Mjpeg),
        _ => None,
    }
}

struct RawFormat {
    fourcc: [u8; 4],
    pixel_format: PixelFormat,
    width: u32,
    height: u32,
    fps: u32,
}

fn enumerate_all_formats(device: &Device) -> Vec<RawFormat> {
    let mut formats = Vec::new();

    let Ok(format_descs) = device.enum_formats() else {
        return formats;
    };

    for desc in format_descs {
        let Some(pixel_format) = fourcc_to_pixel_format(&desc.fourcc.repr) else {
            continue;
        };

        let Ok(frame_sizes) = device.enum_framesizes(desc.fourcc) else {
            continue;
        };

        for frame_size in frame_sizes {
            // For discrete sizes, use the exact value.
            // For stepwise, sample the min and max to cover both extremes.
            let sizes: Vec<(u32, u32)> = match frame_size.size {
                FrameSizeEnum::Discrete(d) => vec![(d.width, d.height)],
                FrameSizeEnum::Stepwise(s) => {
                    let mut sizes = vec![(s.max_width, s.max_height)];
                    if s.min_width != s.max_width || s.min_height != s.max_height {
                        sizes.push((s.min_width, s.min_height));
                    }
                    sizes
                }
            };

            for (width, height) in sizes {
                let Ok(intervals) = device.enum_frameintervals(desc.fourcc, width, height) else {
                    continue;
                };

                for interval in intervals {
                    let fps = match interval.interval {
                        FrameIntervalEnum::Discrete(frac) => {
                            if frac.numerator > 0 {
                                frac.denominator / frac.numerator
                            } else {
                                0
                            }
                        }
                        FrameIntervalEnum::Stepwise(s) => {
                            // Min interval = shortest time per frame = highest fps.
                            if s.min.numerator > 0 {
                                s.min.denominator / s.min.numerator
                            } else {
                                0
                            }
                        }
                    };

                    if fps > 0 {
                        formats.push(RawFormat {
                            fourcc: desc.fourcc.repr,
                            pixel_format,
                            width,
                            height,
                            fps,
                        });
                    }
                }
            }
        }
    }

    formats
}

fn curate_top_formats(device: &Device, index: u8) -> Vec<V4lSource> {
    let all_formats = enumerate_all_formats(device);
    if all_formats.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    // Mjpeg > Raw > Grey
    for pixel_format in [PixelFormat::Mjpeg, PixelFormat::Raw, PixelFormat::Grey] {
        let formats: Vec<_> = all_formats
            .iter()
            .filter(|f| f.pixel_format == pixel_format)
            .collect();

        if formats.is_empty() {
            continue;
        }

        // Highest resolution, tie-break by highest fps.
        if let Some(best_res) = formats
            .iter()
            .max_by_key(|f| (u64::from(f.width) * u64::from(f.height), u64::from(f.fps)))
        {
            let source = V4lSource {
                index,
                format: CameraFormat {
                    width: best_res.width,
                    height: best_res.height,
                    fps: best_res.fps,
                    pixel_format,
                },
                fourcc: best_res.fourcc,
            };

            if !has_same_specs(&candidates, &source) {
                candidates.push(source);
            }
        }

        // Highest framerate, tie-break by highest resolution.
        if let Some(best_fps) = formats
            .iter()
            .max_by_key(|f| (u64::from(f.fps), u64::from(f.width) * u64::from(f.height)))
        {
            let source = V4lSource {
                index,
                format: CameraFormat {
                    width: best_fps.width,
                    height: best_fps.height,
                    fps: best_fps.fps,
                    pixel_format,
                },
                fourcc: best_fps.fourcc,
            };

            if !has_same_specs(&candidates, &source) {
                candidates.push(source);
            }
        }
    }

    candidates
}

fn has_same_specs(candidates: &[V4lSource], source: &V4lSource) -> bool {
    candidates.iter().any(|c| {
        c.format.width == source.format.width
            && c.format.height == source.format.height
            && c.format.fps == source.format.fps
    })
}

/// Queries the system for available cameras and their top format candidates.
///
/// Returns a list of [`CameraInfo`] entries. Each physical camera may appear
/// multiple times — once per curated format (e.g. highest resolution MJPEG,
/// highest framerate raw, etc.).
///
/// This will only work on Linux.
pub fn query_cameras() -> Vec<CameraInfo> {
    let mut cameras = Vec::new();

    for index in 0..64u8 {
        let Ok(device) = Device::new(index as usize) else {
            continue;
        };

        let Ok(caps) = device.query_caps() else {
            continue;
        };

        if !caps.capabilities.contains(Flags::VIDEO_CAPTURE) {
            continue;
        }

        let candidates = curate_top_formats(&device, index);

        for source in candidates {
            cameras.push(CameraInfo {
                name: caps.card.clone(),
                source: CameraSource::V4l(source),
            });
        }
    }

    cameras
}
