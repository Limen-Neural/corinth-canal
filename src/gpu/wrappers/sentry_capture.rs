// ════════════════════════════════════════════════════════════════════
//  gpu/sentry_capture.rs — Sentry telemetry for CUDA launch failures
//
//  Captures detailed context when CUDA kernel launches fail, including:
//  - Kernel name and launch type (PTX/fatbin or C ABI shim)
//  - Grid and block dimensions
//  - Shared memory allocation
//  - CUDA error codes and messages
//  - JIT compilation logs (when applicable)
//
//  Events are sent to Sentry only after the caller initializes a Sentry client
//  (for example via examples/support/observability::init_sentry).
//  This module has no effect when Sentry is disabled.
// ════════════════════════════════════════════════════════════════════

use super::error::GpuError;
use cust::context::legacy::CurrentContext;
use cust::device::DeviceAttribute;
use serde_json::json;

fn runtime_gpu_arch_tag() -> String {
    CurrentContext::get_device()
        .ok()
        .and_then(|device| {
            let major = device
                .get_attribute(DeviceAttribute::ComputeCapabilityMajor)
                .ok()? as u32;
            let minor = device
                .get_attribute(DeviceAttribute::ComputeCapabilityMinor)
                .ok()? as u32;
            Some(format!("sm_{}{}", major, minor))
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Launch type discriminator for Sentry event tagging.
#[derive(Debug, Clone, Copy)]
pub enum LaunchType {
    /// PTX/fatbin kernel launched via cust::launch! macro
    PtxFatbin,
    /// Blackwell-critical F16 kernel launched via C ABI shim
    CAbiShim,
}

impl LaunchType {
    fn as_str(&self) -> &'static str {
        match self {
            LaunchType::PtxFatbin => "ptx_fatbin",
            LaunchType::CAbiShim => "c_abi_shim",
        }
    }
}

/// Structured context for a CUDA kernel launch.
///
/// Captures all relevant metadata needed to diagnose launch failures
/// in Sentry dashboards.
#[derive(Debug, Clone)]
pub struct LaunchContext {
    /// Kernel function name (e.g. "gif_step_weighted", "lif_step")
    pub kernel_name: String,
    /// Launch mechanism (PTX/fatbin or C ABI shim)
    pub launch_type: LaunchType,
    /// Grid dimensions (x, y, z)
    pub grid: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block: (u32, u32, u32),
    /// Shared memory allocation in bytes
    pub shared_mem: u32,
    /// Neuron count (when applicable for temporal kernels)
    pub neuron_count: Option<usize>,
}

/// Capture a CUDA kernel launch failure and send it to Sentry.
///
/// This function is a no-op when no Sentry client is active. Callers are
/// expected to initialize Sentry separately; setting `SENTRY_DSN` alone does
/// not activate capture.
///
/// # Arguments
///
/// * `context` - Structured launch metadata
/// * `error` - The GpuError that occurred
/// * `jit_logs` - Optional (error_log, info_log) tuple from JIT compilation
pub fn capture_launch_failure(
    context: LaunchContext,
    error: &GpuError,
    jit_logs: Option<(String, String)>,
) {
    // Early return if Sentry is not active
    if !sentry::Hub::with_active(|hub| hub.client().is_some()) {
        return;
    }

    let error_category = match error {
        GpuError::LaunchFailed(_) => "launch_failed",
        GpuError::ModuleLoadFailed(_) => "module_load_failed",
        GpuError::MemoryError(_) => "memory_error",
        GpuError::InitFailed(_) => "init_failed",
        GpuError::KernelNotFound(_) => "kernel_not_found",
        GpuError::CudaError(_) => "cuda_error",
        GpuError::NoGpu => "no_gpu",
    };

    let event_message = match error {
        GpuError::ModuleLoadFailed(_) => {
            format!("CUDA fatbin module load failed: {}", context.kernel_name)
        }
        GpuError::LaunchFailed(_) => format!("CUDA kernel launch failed: {}", context.kernel_name),
        _ => format!(
            "CUDA GPU error [{}]: {}",
            error_category, context.kernel_name
        ),
    };

    let gpu_arch = runtime_gpu_arch_tag();

    sentry::with_scope(
        |scope| {
            // Tags for filtering and grouping in Sentry
            scope.set_tag("kernel_name", context.kernel_name.clone());
            scope.set_tag("launch_type", context.launch_type.as_str());
            scope.set_tag("cuda_error_category", error_category);
            scope.set_tag("gpu_arch", gpu_arch.as_str());

            // Structured extras for detailed diagnostics
            scope.set_extra(
                "grid_dimensions",
                json!({
                    "x": context.grid.0,
                    "y": context.grid.1,
                    "z": context.grid.2,
                }),
            );
            scope.set_extra(
                "block_dimensions",
                json!({
                    "x": context.block.0,
                    "y": context.block.1,
                    "z": context.block.2,
                }),
            );
            scope.set_extra("shared_memory_bytes", json!(context.shared_mem));

            if let Some(neuron_count) = context.neuron_count {
                scope.set_extra("neuron_count", json!(neuron_count));
            }

            // Extract CUDA error details from the error message
            let error_message = error.to_string();
            scope.set_extra("cuda_error_message", json!(error_message));

            // Include JIT logs if available (module load failures)
            if let Some((error_log, info_log)) = jit_logs {
                if !error_log.is_empty() {
                    scope.set_extra("jit_error_log", json!(error_log));
                }
                if !info_log.is_empty() {
                    scope.set_extra("jit_info_log", json!(info_log));
                }
            }

            // Set fingerprint for intelligent grouping
            scope.set_fingerprint(Some(&[context.kernel_name.as_str(), error_category]));
        },
        || {
            let msg = event_message.as_str();
            sentry::capture_message(msg, sentry::Level::Error);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_context_construction() {
        let context = LaunchContext {
            kernel_name: "test_kernel".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (8, 1, 1),
            block: (256, 1, 1),
            shared_mem: 4096,
            neuron_count: Some(2048),
        };

        assert_eq!(context.kernel_name, "test_kernel");
        assert_eq!(context.launch_type.as_str(), "ptx_fatbin");
        assert_eq!(context.grid, (8, 1, 1));
        assert_eq!(context.block, (256, 1, 1));
        assert_eq!(context.shared_mem, 4096);
        assert_eq!(context.neuron_count, Some(2048));
    }

    #[test]
    fn launch_type_string_conversion() {
        assert_eq!(LaunchType::PtxFatbin.as_str(), "ptx_fatbin");
        assert_eq!(LaunchType::CAbiShim.as_str(), "c_abi_shim");
    }

    #[test]
    fn capture_launch_failure_without_sentry_is_safe() {
        // This should not panic even when Sentry is not initialized
        let context = LaunchContext {
            kernel_name: "test_kernel".to_string(),
            launch_type: LaunchType::PtxFatbin,
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
            neuron_count: None,
        };
        let error = GpuError::LaunchFailed("test error".to_string());
        capture_launch_failure(context, &error, None);
    }
}
