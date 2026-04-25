mod eye;
mod face;

use std::{marker::PhantomData, ops::Index};

pub use eye::{EyeCalibrator, EyeShape};
pub use face::{FaceShape, ManualFaceCalibrator};

#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Bounds {
    pub min: f32,
    pub max: f32,
    pub lower: f32,
    pub upper: f32,
}

impl Bounds {
    pub(crate) const fn new() -> Self {
        Self {
            min: 0.,
            max: 0.,
            lower: 0.,
            upper: 0.,
        }
    }

    pub(crate) const fn new_01() -> Self {
        Self {
            min: 0.,
            max: 1.,
            lower: 0.,
            upper: 1.,
        }
    }

    pub(crate) const fn new_11() -> Self {
        Self {
            min: -1.,
            max: 1.,
            lower: -1.,
            upper: 1.,
        }
    }

    pub(crate) const fn remap(&self, value: f32) -> f32 {
        self.min + (value - self.lower) * (self.max - self.min) / (self.upper - self.lower)
    }
}

pub trait Shape: Copy + Into<usize> + From<usize> {
    fn count() -> usize;

    fn iter() -> impl Iterator<Item = Self> {
        (0..Self::count()).map(Self::from)
    }
}

#[derive(Copy, Clone)]
pub struct Weights<'a, S> {
    inner: &'a [f32],
    _phantom: PhantomData<S>,
}

impl<'a, S> Weights<'a, S> {
    pub(crate) fn new(inner: &'a [f32]) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub fn as_slice(self) -> &'a [f32] {
        self.inner
    }

    pub fn as_ptr(self) -> *const f32 {
        self.inner.as_ptr()
    }
}

impl<S: Shape> Weights<'_, S> {
    pub fn iter(&self) -> impl Iterator<Item = (S, f32)> + '_ {
        S::iter().map(|s| (s, self.inner[s.into()]))
    }
}

impl<S: Shape> AsRef<[f32]> for Weights<'_, S> {
    fn as_ref(&self) -> &[f32] {
        self.inner
    }
}

impl<S: Shape> Index<S> for Weights<'_, S> {
    type Output = f32;

    fn index(&self, index: S) -> &Self::Output {
        &self.inner[index.into()]
    }
}
