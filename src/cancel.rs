use std::sync::atomic::{AtomicBool, Ordering};

pub trait Cancellation {
    fn is_cancelled(&self) -> bool;
}

pub struct Cancel(AtomicBool);

impl Cancel {
    pub const fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    pub const fn never() -> &'static Self {
        static NEVER: Cancel = Cancel::new();
        &NEVER
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl Default for Cancel {
    fn default() -> Self {
        Self::new()
    }
}

impl Cancellation for Cancel {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}
