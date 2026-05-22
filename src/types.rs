//! Public data types for `corinth-canal`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Dimensionality of the dense embedding the projector hands to OlmoeRouter.
pub const EMBEDDING_DIM: usize = 2048;

/// Supported GGUF model families for the router bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelFamily {
    #[default]
    Olmoe,
    Qwen3Moe,
    Gemma4,
    DeepSeek2,
    LlamaMoe,
    Zaya,
    Glm4,
}

impl ModelFamily {
    pub fn slug(self) -> &'static str {
        match self {
            Self::Olmoe => "olmoe",
            Self::Qwen3Moe => "qwen3_moe",
            Self::Gemma4 => "gemma4",
            Self::DeepSeek2 => "deepseek2",
            Self::LlamaMoe => "llama_moe",
            Self::Zaya => "zaya",
            Self::Glm4 => "glm4",
        }
    }
}

/// Deterministic pulse configuration used to perturb telemetry during validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeartbeatConfig {
    pub enabled: bool,
    pub amplitude: f32,
    pub period_ticks: usize,
    pub duty_cycle: f32,
    pub phase_offset_ticks: usize,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: 0.0,
            period_ticks: 64,
            duty_cycle: 0.25,
            phase_offset_ticks: 0,
        }
    }
}

/// Minimal local telemetry payload used to seed deterministic spike patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub gpu_temp_c: f32,
    pub gpu_power_w: f32,
    pub cpu_tctl_c: f32,
    pub cpu_package_power_w: f32,
    #[serde(default)]
    pub heartbeat_signal: f32,
    #[serde(default)]
    pub heartbeat_enabled: bool,
    pub timestamp_ms: u64,
}

impl TelemetrySnapshot {
    pub fn thermal_stress(&self) -> f32 {
        ((self.gpu_temp_c - 60.0) / 30.0).clamp(0.0, 1.0)
    }
}

/// Top-level configuration for the hybrid quantization pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub gguf_checkpoint_path: String,
    pub model_family: Option<ModelFamily>,
    pub gpu_synapse_tensor_name: String,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub routing_mode: RoutingMode,
    pub snn_steps: usize,
    pub projection_mode: ProjectionMode,
    pub heartbeat: HeartbeatConfig,
    /// Destination path for the GPU routing telemetry CSV written by
    /// `Model::forward_gpu_temporal` (and `Model::forward` on the GPU path).
    /// When `None`, the runtime falls back to the legacy CWD-relative
    /// filename `snn_gpu_routing_telemetry.csv`. Prefer an absolute path
    /// anchored in the caller's per-run artifact directory.
    #[serde(default)]
    pub gpu_routing_telemetry_path: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            gguf_checkpoint_path: String::new(),
            model_family: None,
            gpu_synapse_tensor_name: String::new(),
            num_experts: 8,
            top_k_experts: 1,
            routing_mode: RoutingMode::SpikingSim,
            snn_steps: 20,
            projection_mode: ProjectionMode::SpikingTernary,
            heartbeat: HeartbeatConfig::default(),
            gpu_routing_telemetry_path: None,
        }
    }
}

/// Strategy used to convert spike activity into an OlmoeRouter embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ProjectionMode {
    RateSum,
    TemporalHistogram,
    MembraneSnapshot,
    #[default]
    SpikingTernary,
}

/// Execution mode used by the OlmoeRouter router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RoutingMode {
    StubUniform,
    DenseSim,
    #[default]
    SpikingSim,
}

/// Output of one `Model::forward` pass.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub spike_train: Vec<Vec<usize>>,
    pub firing_rates: Vec<f32>,
    pub membrane_potentials: Vec<f32>,
    pub embedding: Vec<f32>,
    pub expert_weights: Option<Vec<f32>>,
    pub selected_experts: Option<Vec<usize>>,
    pub reasoning: Option<String>,
}

// ── Cloud model metadata ──────────────────────────────────────────────────

/// Execution target for a model in the SAAQ lineup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTarget {
    Local,
    Cloud,
}

/// Model architecture classification for lineup metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitectureClass {
    Dense,
    Moe,
}

/// Metadata stub for a cloud-hosted model that cannot be executed locally.
///
/// Cloud models delegate execution to Dioscuri-Cloud. corinth-canal is
/// responsible for recording the candidate in experiment manifests and
/// fail-fast behaviour when the required cloud provider env vars are unset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudModelSpec {
    /// Directory-safe identifier used in artifact paths.
    pub slug: String,
    /// Model family for routing tensor selection.
    pub family: Option<ModelFamily>,
    /// Provider / model ID (e.g. `nvidia/nemotron-nano-4b`).
    pub cloud_model_id: String,
    /// Canonical source URL (model card or provider listing).
    pub source_url: String,
    /// Execution target.
    pub target: ModelTarget,
    /// Architecture class.
    pub architecture: ModelArchitectureClass,
    /// Known active parameter count (e.g. `"2.4B"`).
    pub active_params: String,
    /// Known total parameter count (e.g. `"8B"`).
    pub total_params: String,
    /// Expected provider / runtime format on the cloud side
    /// (e.g. `"nvcf-nim"`, `"openai-compat"`, `"fp8-safetensors"`).
    pub provider_format: String,
    /// Environment variable names required for cloud execution.
    /// corinth-canal checks these at startup; if any are unset, execution
    /// fails fast with a diagnostic message. Values never appear in artifacts.
    #[serde(default)]
    pub required_env_vars: Vec<String>,
}

impl CloudModelSpec {
    /// Returns `true` when every env var in `required_env_vars` is set
    /// to a non-empty string.
    pub fn cloud_provider_available(&self) -> bool {
        self.required_env_vars
            .iter()
            .all(|var| std::env::var(var).is_ok_and(|v| !v.is_empty()))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CloudModelSpec, ModelArchitectureClass, ModelFamily, ModelTarget,
    };

    #[test]
    fn model_family_slug_covers_new_variants() {
        assert_eq!(ModelFamily::Zaya.slug(), "zaya");
        assert_eq!(ModelFamily::Glm4.slug(), "glm4");
    }

    #[test]
    fn cloud_model_spec_provider_availability_checks_env_vars() {
        let missing = "CORINTH_CANAL_TEST_MISSING_PROVIDER_VAR";

        let spec = CloudModelSpec {
            slug: "test-cloud".into(),
            family: Some(ModelFamily::Zaya),
            cloud_model_id: "provider/test-cloud".into(),
            source_url: "https://example.invalid/test-cloud".into(),
            target: ModelTarget::Cloud,
            architecture: ModelArchitectureClass::Moe,
            active_params: "1B".into(),
            total_params: "8B".into(),
            provider_format: "openai-compat".into(),
            required_env_vars: vec!["PATH".into(), missing.into()],
        };
        assert!(!spec.cloud_provider_available());

        let local_dense = CloudModelSpec {
            slug: "test-local".into(),
            family: Some(ModelFamily::Glm4),
            cloud_model_id: "provider/test-local".into(),
            source_url: "https://example.invalid/test-local".into(),
            target: ModelTarget::Local,
            architecture: ModelArchitectureClass::Dense,
            active_params: "3B".into(),
            total_params: "3B".into(),
            provider_format: "nvcf-nim".into(),
            required_env_vars: vec!["PATH".into()],
        };
        assert!(local_dense.cloud_provider_available());
    }
}
