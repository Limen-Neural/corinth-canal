// ════════════════════════════════════════════════════════════════════
//  gpu/mod.rs — GPU sub-module declarations and public re-exports
// ════════════════════════════════════════════════════════════════════

pub mod accelerator;
pub mod context;
pub mod error;
mod ffi;
pub mod kernel;
pub mod memory;
mod sentry_capture;

pub use accelerator::GpuAccelerator;
pub use context::GpuContext;
pub use error::{GpuError, GpuResult};
pub use kernel::KernelModule;
pub use memory::GpuBuffer;
pub(crate) use sentry_capture::{LaunchContext, LaunchType, capture_launch_failure};
