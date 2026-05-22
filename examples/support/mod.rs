//! Shared helper functions for the example binaries.

pub mod config;
pub mod observability;

#[allow(unused_imports)]
pub use config::RunConfig;

use corinth_canal::{
    HeartbeatConfig, ModelFamily, SaaqUpdateRule, moe::OlmoeRouter, moe::RoutingMode,
};
#[cfg(feature = "cuda")]
use corinth_canal::{model::ModelConfig, projector::ProjectionMode};
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_MATH_PROMPT_TEXT: &str = "The derivative of a constant is mathematically zero.";

pub const DEFAULT_RUST_SYNTAX_PROMPT_TEXT: &str =
    "fn main() { println!(\"Hello from a spiking MoE model.\"); }";

pub const DEFAULT_ENGLISH_SNN_PROMPT_TEXT: &str = "Let's teach this MoE model about SNN.";

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ValidationModelSpec {
    pub slug: String,
    pub family: Option<ModelFamily>,
    pub path: String,
    /// Optional per-model routing mode override. Set by lineup-config entries
    /// (`configs/saaq15_moe_lineup.toml`); autodiscovered / CLI-injected
    /// specs leave this `None` and fall back to `ModelConfig::routing_mode`.
    pub routing_mode: Option<RoutingMode>,
}

#[allow(dead_code)]
#[cfg(feature = "cuda")]
pub fn default_spiking_model_config(gguf_checkpoint_path: String, snn_steps: usize) -> ModelConfig {
    let probe = if gguf_checkpoint_path.trim().is_empty() {
        None
    } else {
        OlmoeRouter::probe_model(&gguf_checkpoint_path, None).ok()
    };

    ModelConfig {
        gguf_checkpoint_path,
        model_family: probe.as_ref().map(|metadata| metadata.family),
        gpu_synapse_tensor_name: probe
            .as_ref()
            .and_then(|metadata| metadata.real_gpu_synapse_tensor_name.clone())
            .unwrap_or_default(),
        num_experts: probe
            .as_ref()
            .map(|metadata| metadata.num_experts)
            .unwrap_or(8),
        top_k_experts: probe
            .as_ref()
            .map(|metadata| metadata.expert_used_count.max(1))
            .unwrap_or(1),
        routing_mode: RoutingMode::SpikingSim,
        snn_steps,
        projection_mode: ProjectionMode::SpikingTernary,
        heartbeat: heartbeat_config_from_env(),
        gpu_routing_telemetry_path: None,
    }
}

pub fn prompt_profile_slug() -> String {
    std::env::var("PROMPT_PROFILE")
        .unwrap_or_else(|_| "math_logic".into())
        .to_ascii_lowercase()
}

pub fn prompt_text_for_profile(profile: &str) -> &'static str {
    match profile {
        "math_logic" | "math" => DEFAULT_MATH_PROMPT_TEXT,
        "rust_syntax" | "rust" => DEFAULT_RUST_SYNTAX_PROMPT_TEXT,
        "english_snn" | "english" | "snn" => DEFAULT_ENGLISH_SNN_PROMPT_TEXT,
        _ => DEFAULT_MATH_PROMPT_TEXT,
    }
}

pub fn model_family_override_from_env() -> Option<ModelFamily> {
    let value = std::env::var("MODEL_FAMILY").ok()?;
    parse_family_slug(&value)
}

/// Shared family-slug parser used by `MODEL_FAMILY` and by lineup-config
/// `family = "..."` entries. Case-insensitive; returns `None` for unknown
/// slugs so callers can treat that as a soft validation error.
pub fn parse_family_slug(value: &str) -> Option<ModelFamily> {
    match value.trim().to_ascii_lowercase().as_str() {
        "olmoe" => Some(ModelFamily::Olmoe),
        "qwen3moe" | "qwen3_moe" | "qwen" => Some(ModelFamily::Qwen3Moe),
        "gemma4" | "gemma_4" | "gemma" => Some(ModelFamily::Gemma4),
        "deepseek2" | "deepseek_v2" | "deepseek" => Some(ModelFamily::DeepSeek2),
        "llama" | "llama_moe" | "llama3_moe" => Some(ModelFamily::LlamaMoe),
        "zaya" | "zaya1" | "zaya1_8b" => Some(ModelFamily::Zaya),
        "glm4" | "glm_4" | "glm4moe" | "glm" => Some(ModelFamily::Glm4),
        _ => None,
    }
}

/// Same precedence rules as `routing_mode_override_from_env` but reading
/// from an arbitrary string (used for lineup-config entries). Returns
/// `None` for unknown values so callers can treat that as a soft validation
/// error without aborting the whole sweep.
pub fn parse_routing_mode(value: &str) -> Option<RoutingMode> {
    match value.trim().to_ascii_lowercase().as_str() {
        "dense" | "dense_sim" => Some(RoutingMode::DenseSim),
        "stub" | "stub_uniform" => Some(RoutingMode::StubUniform),
        "spiking" | "spiking_sim" => Some(RoutingMode::SpikingSim),
        _ => None,
    }
}

// ── Cloud model metadata ──────────────────────────────────────────────────

/// Parsed representation of a cloud model entry from
/// `configs/saaq15_cloud_lineup.toml`.
#[derive(Debug, Clone)]
pub struct CloudModelEntry {
    pub slug: String,
    pub family: Option<ModelFamily>,
    pub cloud_model_id: String,
    pub source_url: String,
    pub architecture: String,
    pub active_params: String,
    pub total_params: String,
    pub provider_format: String,
    pub required_env_vars: Vec<String>,
    /// Whether all `required_env_vars` are set to non-empty values.
    pub provider_available: bool,
}

/// Parse a `configs/saaq15_cloud_lineup.toml` file and return every entry.
///
/// Unknown families are reported via stderr but accepted (family is left
/// `None` for probe-based inference downstream).
///
/// ---
/// *Goose agent (deepseek-v4-pro model) — parsing cloud lineup for
/// Dioscuri-Cloud delegation metadata.*
pub fn load_cloud_lineup(path: &Path) -> Result<Vec<CloudModelEntry>, Box<dyn std::error::Error>> {
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawCloudLineup {
        #[serde(default)]
        model: Vec<RawCloudModel>,
    }

    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawCloudModel {
        slug: String,
        #[serde(default)]
        family: String,
        cloud_model_id: String,
        source_url: String,
        target: String,
        architecture: String,
        active_params: String,
        total_params: String,
        provider_format: String,
        #[serde(default)]
        required_env_vars: Vec<String>,
    }

    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: RawCloudLineup =
        toml::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?;

    let mut out = Vec::with_capacity(parsed.model.len());
    for entry in parsed.model {
        let target = entry.target.trim().to_ascii_lowercase();
        if target != "cloud" {
            return Err(format!(
                "cloud_lineup: invalid target for slug={}: expected \"cloud\", got \"{}\"",
                entry.slug, entry.target
            )
            .into());
        }
        let architecture = entry.architecture.trim().to_ascii_lowercase();
        if architecture != "moe" && architecture != "dense" {
            return Err(format!(
                "cloud_lineup: invalid architecture for slug={}: expected \"moe\" or \"dense\", got \"{}\"",
                entry.slug, entry.architecture
            )
            .into());
        }
        let family = parse_family_slug(&entry.family);
        if family.is_none() && !entry.family.is_empty() {
            eprintln!(
                "cloud_lineup: unknown family '{}' for slug={}; leaving family inference to probe",
                entry.family, entry.slug
            );
        }
        let provider_available = entry
            .required_env_vars
            .iter()
            .all(|var| std::env::var(var).is_ok_and(|v| !v.is_empty()));

        if !provider_available {
            let unset: Vec<_> = entry
                .required_env_vars
                .iter()
                .filter(|var| !std::env::var(var).is_ok_and(|v| !v.is_empty()))
                .collect();
            eprintln!(
                "cloud_lineup: provider unavailable for slug={} ({}): missing env vars: {}",
                entry.slug,
                entry.cloud_model_id,
                unset
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        out.push(CloudModelEntry {
            slug: entry.slug,
            family,
            cloud_model_id: entry.cloud_model_id,
            source_url: entry.source_url,
            architecture,
            active_params: entry.active_params,
            total_params: entry.total_params,
            provider_format: entry.provider_format,
            required_env_vars: entry.required_env_vars,
            provider_available,
        });
    }
    Ok(out)
}

/// Fail-fast guard for cloud model execution.
///
/// Re-checks `required_env_vars` at call time and returns an error
/// describing any that are missing. Callers in the SAAQ runner should check
/// this before attempting any cloud-backed forward passes so that the
/// artifact manifest can record the skip reason.
///
/// corinth-canal does not create cloud resources — it delegates execution
/// to Dioscuri-Cloud. This guard only validates that the required env vars
/// are present.
///
/// ---
/// *Goose agent (deepseek-v4-pro model) — implementing cloud execution guard
/// for fail-fast behavior on missing provider configuration.*
pub fn cloud_execution_guard(entry: &CloudModelEntry) -> Result<(), String> {
    let unset: Vec<_> = entry
        .required_env_vars
        .iter()
        .filter(|var| !std::env::var(var).is_ok_and(|v| !v.is_empty()))
        .collect();
    if unset.is_empty() {
        return Ok(());
    }
    Err(format!(
        "cloud model '{}' ({}) cannot execute: required env vars not set: {}. \
         Cloud execution is delegated to Dioscuri-Cloud; configure these env \
         vars in your deployment environment.",
        entry.slug,
        entry.cloud_model_id,
        unset
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Load the cloud lineup path from the `CLOUD_LINEUP_CONFIG` env var.
/// Returns `None` when unset or empty.
pub fn cloud_lineup_path_from_env() -> Option<PathBuf> {
    std::env::var("CLOUD_LINEUP_CONFIG")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

// ── Safetensors manifest lineup ───────────────────────────────────────────

/// Parsed representation of a safetensors model entry from
/// `configs/safetensors_lineup.toml`.
#[derive(Debug, Clone)]
pub struct SafetensorsModelEntry {
    pub slug: String,
    pub family: Option<ModelFamily>,
    pub path: PathBuf,
    pub target: String,
}

/// Parse a `configs/safetensors_lineup.toml` file and return every entry
/// whose path exists on disk. Missing paths are skipped with a warning.
///
/// ---
/// *Goose agent (deepseek-v4-pro model) — adding safetensors lineup parser
/// for batch manifest generation of local safetensors checkpoints.*
pub fn load_safetensors_lineup(
    path: &Path,
) -> Result<Vec<SafetensorsModelEntry>, Box<dyn std::error::Error>> {
    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawSafetensorsLineup {
        #[serde(default)]
        model: Vec<RawSafetensorsModel>,
    }

    #[derive(Debug, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawSafetensorsModel {
        slug: String,
        #[serde(default)]
        family: String,
        path: String,
        target: String,
    }

    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: RawSafetensorsLineup =
        toml::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?;

    let mut out = Vec::with_capacity(parsed.model.len());
    for entry in parsed.model {
        if !entry.target.trim().eq_ignore_ascii_case("local") {
            return Err(format!(
                "safetensors_lineup: invalid target for slug={}: expected \"local\", got \"{}\"",
                entry.slug, entry.target
            )
            .into());
        }
        let trimmed_path = entry.path.trim();
        let entry_path = PathBuf::from(trimmed_path);
        if !entry_path.exists() {
            eprintln!(
                "safetensors_lineup: skipping slug={}: path not found: {}",
                entry.slug, trimmed_path
            );
            continue;
        }
        let family = parse_family_slug(&entry.family);
        if family.is_none() && !entry.family.is_empty() {
            eprintln!(
                "safetensors_lineup: unknown family '{}' for slug={}",
                entry.family, entry.slug
            );
        }
        out.push(SafetensorsModelEntry {
            slug: entry.slug,
            family,
            path: entry_path,
            target: entry.target,
        });
    }
    Ok(out)
}

/// Load the safetensors lineup path from the `SAFETENSORS_LINEUP_CONFIG`
/// env var. Returns `None` when unset or empty.
pub fn safetensors_lineup_path_from_env() -> Option<PathBuf> {
    std::env::var("SAFETENSORS_LINEUP_CONFIG")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
}

pub fn saaq_update_rule_from_env() -> SaaqUpdateRule {
    match std::env::var("SAAQ_RULE")
        .unwrap_or_else(|_| "saaq_v1_5".into())
        .to_ascii_lowercase()
        .as_str()
    {
        "legacy" | "legacy_v1_0" | "v1_0" | "saaq_v1_0" => SaaqUpdateRule::LegacyV1_0,
        _ => SaaqUpdateRule::SaaqV1_5SqrtRate,
    }
}

#[allow(dead_code)]
pub fn heartbeat_config_from_env() -> HeartbeatConfig {
    HeartbeatConfig {
        enabled: env_flag("HEARTBEAT_ENABLED", false),
        amplitude: env_f32("HEARTBEAT_AMPLITUDE", 0.85),
        period_ticks: env_usize("HEARTBEAT_PERIOD_TICKS", 48),
        duty_cycle: env_f32("HEARTBEAT_DUTY_CYCLE", 0.25),
        phase_offset_ticks: env_usize("HEARTBEAT_PHASE_OFFSET_TICKS", 0),
    }
}

pub fn heartbeat_modes_for_matrix() -> Vec<bool> {
    if let Ok(value) = std::env::var("HEARTBEAT_MATRIX") {
        let lower = value.to_ascii_lowercase();
        if lower == "off" {
            return vec![false];
        }
        if lower == "on" {
            return vec![true];
        }
    }
    vec![false, true]
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptEmbeddingProvider {
    Ollama,
    SyntheticFallback,
}

fn resolve_prompt_embedding_provider(value: Option<&str>) -> PromptEmbeddingProvider {
    match value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_ascii_lowercase())
        .as_deref()
    {
        None | Some("ollama") => PromptEmbeddingProvider::Ollama,
        _ => PromptEmbeddingProvider::SyntheticFallback,
    }
}

#[allow(dead_code)]
pub fn prompt_embedding_for_validation(
    prompt_text: &str,
    target_dim: usize,
) -> Result<(Vec<f32>, String), Box<dyn std::error::Error>> {
    let provider = std::env::var("EMBEDDING_PROVIDER").ok();

    if resolve_prompt_embedding_provider(provider.as_deref()) == PromptEmbeddingProvider::Ollama {
        match pooled_prompt_embedding_from_ollama(prompt_text, target_dim) {
            Ok((embedding, label)) => return Ok((embedding, label)),
            Err(error) => {
                eprintln!(
                    "Ollama prompt embedding unavailable: {}. Falling back to deterministic text hash embedding.",
                    error
                );
            }
        }
    } else {
        eprintln!(
            "Unknown embedding provider '{}'. Falling back to deterministic text hash embedding.",
            provider.as_deref().unwrap_or("<unset>")
        );
    }

    Ok((
        synthetic_text_embedding(prompt_text, target_dim),
        "text_hash_fallback".into(),
    ))
}

#[allow(dead_code)]
pub fn pooled_prompt_embedding_from_ollama(
    prompt_text: &str,
    target_dim: usize,
) -> Result<(Vec<f32>, String), Box<dyn std::error::Error>> {
    let model =
        std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    let url = std::env::var("OLLAMA_EMBED_URL")
        .unwrap_or_else(|_| "http://localhost:11434/api/embed".to_string());
    let prefix =
        std::env::var("OLLAMA_EMBED_PREFIX").unwrap_or_else(|_| "classification: ".to_string());

    let input = format!("{}{}", prefix, prompt_text);
    let payload = serde_json::json!({
        "model": model,
        "input": input,
    });
    let payload_str = serde_json::to_string(&payload)?;

    let output = Command::new("curl")
        .arg("--fail-with-body")
        .arg("--silent")
        .arg("--show-error")
        .arg("--connect-timeout")
        .arg("5")
        .arg("--max-time")
        .arg("30")
        .arg("-X")
        .arg("POST")
        .arg(&url)
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(&payload_str)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = match (stderr.trim(), stdout.trim()) {
            ("", "") => format!("exit status {}", output.status),
            (stderr, "") => stderr.to_owned(),
            ("", stdout) => stdout.to_owned(),
            (stderr, stdout) => format!("{stderr}; response: {stdout}"),
        };
        return Err(Error::other(format!("Ollama curl request failed: {detail}")).into());
    }

    let stdout = String::from_utf8(output.stdout)?;

    #[derive(serde::Deserialize)]
    struct OllamaResponse {
        embeddings: Option<Vec<Vec<f32>>>,
    }

    let response: OllamaResponse = serde_json::from_str(&stdout).map_err(|e| {
        Error::other(format!(
            "Failed to parse Ollama response: {e}\nResponse: {stdout}"
        ))
    })?;

    let mut embedding = response
        .embeddings
        .and_then(|mut embs| embs.pop())
        .ok_or_else(|| Error::other("Ollama response did not contain embeddings"))?;

    if embedding.len() != target_dim {
        embedding = resample_embedding(&embedding, target_dim);
    }
    normalize_embedding(&mut embedding);

    let label = format!("ollama:{}", model);
    Ok((embedding, label))
}

pub fn discover_validation_models() -> Vec<ValidationModelSpec> {
    if let Ok(path) = std::env::var("GGUF_CHECKPOINT_PATH")
        && !path.trim().is_empty()
    {
        let family = OlmoeRouter::probe_model(&path, None)
            .ok()
            .map(|metadata| metadata.family);
        return vec![ValidationModelSpec {
            slug: slug_from_path(&path),
            family,
            path,
            routing_mode: None,
        }];
    }

    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let root = PathBuf::from(home)
        .join("Downloads")
        .join("SNN_Quantization");
    let candidates = [
        (
            "olmoe_baseline",
            Some(ModelFamily::Olmoe),
            PathBuf::from("olmoe-0125-gguf/OLMoE-1B-7B-0125-Instruct-F16.gguf"),
        ),
        (
            "qwen3_moe_i1_iq3_m",
            Some(ModelFamily::Qwen3Moe),
            PathBuf::from("models/qwen3-moe-i1-GGUF/qwen3-moe.i1-IQ3_M.gguf"),
        ),
        (
            "gemma4_26b_a4b_iq4_nl",
            Some(ModelFamily::Gemma4),
            PathBuf::from("models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf"),
        ),
        (
            "deepseek_coder_v2_lite_q6_k_l",
            Some(ModelFamily::DeepSeek2),
            PathBuf::from(
                "models/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q6_K_L.gguf",
            ),
        ),
        (
            "llama_3_2_dark_champion_q5_k_m",
            Some(ModelFamily::LlamaMoe),
            PathBuf::from(
                "models/Llama-3.2-8X3B-MOE-Dark-Champion-GGUF/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-q5_k_m.gguf",
            ),
        ),
        (
            "zaya1_8b_q8_0",
            Some(ModelFamily::Zaya),
            PathBuf::from("models/ZAYA1-8B-GGUF/ZAYA1-8B-Q8_0.gguf"),
        ),
        (
            "glm46v_flash_q8_0",
            Some(ModelFamily::Glm4),
            PathBuf::from("models/GLM-4.6V-Flash-GGUF_Q8_0/GLM-4.6V-Flash-Q8_0.gguf"),
        ),
        (
            "kimi_vl_a3b_q6_k",
            Some(ModelFamily::DeepSeek2),
            PathBuf::from("models/Kimi-VL-A3B-Instruct-GGUF_Q6_K/Kimi-VL-A3B-Instruct-Q6_K.gguf"),
        ),
        (
            "marco_nano_base_q8_0",
            Some(ModelFamily::Qwen3Moe),
            PathBuf::from("models/Marco-Nano-Base-GGUF_Q8_0/Marco-Nano-Base.Q8_0.gguf"),
        ),
    ];

    candidates
        .into_iter()
        .filter_map(|(slug, family, rel)| {
            let path = root.join(rel);
            path.exists().then(|| ValidationModelSpec {
                slug: slug.into(),
                family,
                path: path.to_string_lossy().into_owned(),
                routing_mode: None,
            })
        })
        .collect()
}

pub fn ticks_from_env(default_ticks: usize) -> usize {
    env_usize("TICKS", default_ticks)
}

pub fn repeat_count_from_env() -> usize {
    env_usize("REPEAT_COUNT", 1).max(1)
}

/// Source of per-tick telemetry snapshots feeding the validation runner.
///
/// Default is [`TelemetrySource::Synthetic`] so a fresh clone never silently
/// depends on a machine-specific CSV path. CSV replay is opt-in via
/// `TELEMETRY_SOURCE=csv` for RE4/Cyberpunk corpus generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetrySource {
    Synthetic,
    Csv,
}

/// Telemetry state resolved once per process and reused by every tick.
///
/// `source_label` is what lands in the directory path and manifest: one of
/// `synthetic`, `synthetic_fallback`, or `csv_<stem>` (e.g. `csv_re4` for
/// `telemetry.csv`). `rows` is only populated on a successful CSV load.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ResolvedTelemetry {
    pub source: TelemetrySource,
    pub source_label: String,
    pub csv_path: Option<PathBuf>,
    pub rows: Option<Vec<corinth_canal::TelemetrySnapshot>>,
}

impl ResolvedTelemetry {
    #[allow(dead_code)]
    pub fn row_count(&self) -> Option<usize> {
        self.rows.as_ref().map(|rows| rows.len())
    }
}

pub fn telemetry_source_from_env() -> TelemetrySource {
    match std::env::var("TELEMETRY_SOURCE")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .trim()
    {
        "csv" => TelemetrySource::Csv,
        // Empty string and any unrecognised value fall back to Synthetic
        // contributors never get surprised by a missing CSV path.
        _ => TelemetrySource::Synthetic,
    }
}

pub fn telemetry_csv_path_from_env() -> PathBuf {
    if let Ok(value) = std::env::var("TELEMETRY_CSV_PATH") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    // Fallback to a repo-relative placeholder or just a generic name.
    // The runner will handle the "file not found" case by falling back
    // to synthetic telemetry.
    PathBuf::from("telemetry.csv")
}

const TELEMETRY_CSV_HEADER: &str =
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

/// Parse a canonical telemetry CSV exported by `gaming-telemetry` into a
/// vector of [`corinth_canal::TelemetrySnapshot`] ready for replay.
///
/// Fails fast on header mismatch; silently skips malformed data rows (counted
/// in the returned log line via stderr) so a few dropped samples don't abort
/// a 2000-row sweep.
pub fn load_csv_telemetry_rows(
    path: &Path,
) -> Result<Vec<corinth_canal::TelemetrySnapshot>, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path).map_err(|error| {
        Error::other(format!(
            "telemetry CSV '{}' could not be read: {error}",
            path.display()
        ))
    })?;

    let mut lines = contents.lines();
    let header = lines
        .next()
        .ok_or_else(|| Error::other(format!("telemetry CSV '{}' is empty", path.display())))?
        .trim();
    if header != TELEMETRY_CSV_HEADER {
        return Err(Error::other(format!(
            "telemetry CSV '{}' header mismatch: expected '{TELEMETRY_CSV_HEADER}', got '{header}'",
            path.display()
        ))
        .into());
    }

    let mut rows = Vec::new();
    let mut skipped = 0usize;
    for (idx, raw_line) in lines.enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 5 {
            skipped += 1;
            continue;
        }
        let parsed = (
            fields[0].parse::<u64>().ok(),
            parse_finite_f32(fields[1]),
            parse_finite_f32(fields[2]),
            parse_finite_f32(fields[3]),
            parse_finite_f32(fields[4]),
        );
        let (
            Some(timestamp_ms),
            Some(gpu_temp_c),
            Some(gpu_power_w),
            Some(cpu_tctl_c),
            Some(cpu_package_power_w),
        ) = parsed
        else {
            skipped += 1;
            let _ = idx;
            continue;
        };
        rows.push(corinth_canal::TelemetrySnapshot {
            timestamp_ms,
            gpu_temp_c,
            gpu_power_w,
            cpu_tctl_c,
            cpu_package_power_w,
            heartbeat_signal: 0.0,
            heartbeat_enabled: false,
        });
    }

    if skipped > 0 {
        eprintln!(
            "load_csv_telemetry_rows: skipped {skipped} malformed row(s) in '{}'",
            path.display()
        );
    }

    Ok(rows)
}

/// Resolve the process-wide telemetry source once.
///
/// For `Csv`, loads and validates the CSV file up front; if the file is
/// missing, malformed, or empty, emits a single warning to stderr and
/// degrades to `Synthetic`, stamping `source_label = "synthetic_fallback"`
/// so the manifest faithfully records what actually happened.
pub fn resolve_telemetry_source() -> ResolvedTelemetry {
    match telemetry_source_from_env() {
        TelemetrySource::Synthetic => ResolvedTelemetry {
            source: TelemetrySource::Synthetic,
            source_label: "synthetic".to_string(),
            csv_path: None,
            rows: None,
        },
        TelemetrySource::Csv => {
            let csv_path = telemetry_csv_path_from_env();
            match load_csv_telemetry_rows(&csv_path) {
                Ok(rows) if !rows.is_empty() => {
                    let label = csv_source_label(&csv_path);
                    ResolvedTelemetry {
                        source: TelemetrySource::Csv,
                        source_label: label,
                        csv_path: Some(csv_path),
                        rows: Some(rows),
                    }
                }
                Ok(_) => {
                    eprintln!(
                        "resolve_telemetry_source: CSV '{}' is empty; falling back to synthetic",
                        csv_path.display()
                    );
                    ResolvedTelemetry {
                        source: TelemetrySource::Synthetic,
                        source_label: "synthetic_fallback".to_string(),
                        csv_path: Some(csv_path),
                        rows: None,
                    }
                }
                Err(error) => {
                    eprintln!(
                        "resolve_telemetry_source: CSV '{}' failed to load: {error}; falling back to synthetic",
                        csv_path.display()
                    );
                    ResolvedTelemetry {
                        source: TelemetrySource::Synthetic,
                        source_label: "synthetic_fallback".to_string(),
                        csv_path: Some(csv_path),
                        rows: None,
                    }
                }
            }
        }
    }
}

/// Convert a CSV path into a directory-safe source slug. The stem
/// `telemetry` is treated as the canonical RE4 corpus and renders as
/// `csv_re4`; any other stem becomes `csv_<stem>`.
fn csv_source_label(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_ascii_lowercase();
    if stem == "telemetry" {
        "csv_re4".to_string()
    } else {
        let sanitized = stem.replace([' ', '.'], "_");
        format!("csv_{sanitized}")
    }
}

/// Produce the telemetry snapshot for a given tick. For CSV replay this
/// wraps around when `tick >= rows.len()`; the caller is responsible for
/// warning when `TICKS > row_count` (see `saaq_latent_calibration`).
///
/// `timestamp_ms` is always rewritten to `tick + 1` so the resulting latent
/// CSV joins 1-to-1 against `tick_telemetry.txt` on tick index regardless
/// of the underlying CSV's absolute timestamps.
#[allow(dead_code)]
pub fn telemetry_snapshot_for_tick(
    tick: usize,
    resolved: &ResolvedTelemetry,
) -> corinth_canal::TelemetrySnapshot {
    let mut snap = match (resolved.source, resolved.rows.as_ref()) {
        (TelemetrySource::Csv, Some(rows)) if !rows.is_empty() => {
            let idx = tick % rows.len();
            rows[idx].clone()
        }
        _ => synthetic_base_snapshot(tick),
    };
    snap.timestamp_ms = (tick as u64) + 1;
    snap
}

fn parse_finite_f32(value: &str) -> Option<f32> {
    let parsed = value.parse::<f32>().ok()?;
    if parsed.is_finite() {
        Some(parsed)
    } else {
        None
    }
}

#[allow(dead_code)]
pub fn synthetic_base_snapshot(tick: usize) -> corinth_canal::TelemetrySnapshot {
    let phase = tick as f32 * 0.041;
    corinth_canal::TelemetrySnapshot {
        gpu_temp_c: 68.0 + phase.sin() * 2.8,
        gpu_power_w: 232.0 + phase.cos() * 11.5,
        cpu_tctl_c: 73.0 + (phase * 0.9).sin() * 2.2,
        cpu_package_power_w: 116.0 + (phase * 1.1).cos() * 7.4,
        heartbeat_signal: 0.0,
        heartbeat_enabled: false,
        timestamp_ms: tick as u64,
    }
}

#[allow(dead_code)]
pub fn heartbeat_gain(signal: f32) -> f32 {
    (1.0 + signal * 0.28).max(0.15)
}

/// Scale factor applied to the prompt embedding before GPU temporal upload.
///
/// L2-normalised 2048-dim embeddings have per-element magnitude ≈ 0.022.
/// The GIF kernel's `GIF_DRIVE_SCALE=0.75` and `GIF_THRESHOLD_BASE=0.65`
/// require effective drive ≥ ~0.87 per tick to fire.  A gain of 32 lifts
/// per-element input from ~0.022 to ~0.7, producing dot-product drives that
/// comfortably cross threshold and yield healthy 5–15 % firing rates.
///
/// Override via `INPUT_DRIVE_GAIN` env var for per-model tuning.
#[allow(dead_code)]
pub fn input_drive_gain_from_env() -> f32 {
    env_f32("INPUT_DRIVE_GAIN", 32.0)
}

/// Parse `ROUTING_MODE` into a [`RoutingMode`]. Returns `None` when the env
/// var is unset so callers can keep the config-provided default.
pub fn routing_mode_override_from_env() -> Option<RoutingMode> {
    let raw = std::env::var("ROUTING_MODE").ok()?;
    match raw.to_ascii_lowercase().as_str() {
        "dense" => Some(RoutingMode::DenseSim),
        "stub" => Some(RoutingMode::StubUniform),
        "spiking" | "spiking_sim" => Some(RoutingMode::SpikingSim),
        _ => None,
    }
}

fn slug_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("gguf_model")
        .replace(['.', '-', ' '], "_")
        .to_ascii_lowercase()
}

#[allow(dead_code)]
fn resample_embedding(input: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if input.len() == target_len {
        return input.to_vec();
    }
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    if target_len == 1 {
        return vec![input.iter().sum::<f32>() / input.len() as f32];
    }

    let scale = (input.len() - 1) as f32 / (target_len - 1) as f32;
    let mut out = Vec::with_capacity(target_len);

    for idx in 0..target_len {
        let source = idx as f32 * scale;
        let lo = source.floor() as usize;
        let hi = source.ceil().min((input.len() - 1) as f32) as usize;
        if lo == hi {
            out.push(input[lo]);
        } else {
            let t = source - lo as f32;
            out.push(input[lo] * (1.0 - t) + input[hi] * t);
        }
    }
    out
}

#[allow(dead_code)]
fn normalize_embedding(values: &mut [f32]) {
    let l2_norm = values.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if l2_norm > 1e-8 {
        for value in values {
            *value /= l2_norm;
        }
    }
}

#[allow(dead_code)]
fn synthetic_text_embedding(prompt_text: &str, target_dim: usize) -> Vec<f32> {
    if target_dim == 0 {
        return Vec::new();
    }

    let bytes = prompt_text.as_bytes();
    if bytes.is_empty() {
        return vec![0.0; target_dim];
    }

    let mut embedding = vec![0.0f32; target_dim];
    for (idx, _) in bytes.iter().enumerate() {
        let start = idx.saturating_sub(3);
        let hash = fnv1a64(&bytes[start..=idx]);
        let slot = (hash as usize) % target_dim;
        let sign = if ((hash >> 11) & 1) == 0 { 1.0 } else { -1.0 };
        let magnitude = 1.0 + (bytes[idx] as f32 / 255.0);
        embedding[slot] += sign * magnitude;
    }

    normalize_embedding(&mut embedding);
    embedding
}

#[allow(dead_code)]
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[allow(dead_code)]
fn env_f32(key: &str, default_value: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default_value)
}

fn env_usize(key: &str, default_value: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default_value)
}

pub(super) fn env_flag(key: &str, default_value: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp_csv(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "corinth_canal_support_{}_{}.csv",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn load_csv_accepts_canonical_header_and_parses_rows() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,60.5,250.0,70.0,120.0\n\
                   2000,61.0,252.5,70.5,121.5\n";
        let path = write_temp_csv("canonical", csv);
        let rows = load_csv_telemetry_rows(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].timestamp_ms, 1000);
        assert!((rows[0].gpu_temp_c - 60.5).abs() < 1e-6);
        assert_eq!(rows[1].timestamp_ms, 2000);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_csv_rejects_bad_header() {
        let csv = "t,gpu,gpuw,cpu,cpuw\n1000,60,250,70,120\n";
        let path = write_temp_csv("bad_header", csv);
        let err = load_csv_telemetry_rows(&path).unwrap_err();
        assert!(err.to_string().contains("header mismatch"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_csv_skips_malformed_rows() {
        let csv = "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w\n\
                   1000,60.5,250.0,70.0,120.0\n\
                   malformed,row\n\
                   2000,NaN,250.0,70.0,120.0\n\
                   3000,61.0,252.5,70.5,121.5\n";
        let path = write_temp_csv("skip_bad", csv);
        let rows = load_csv_telemetry_rows(&path).unwrap();
        assert_eq!(rows.len(), 2, "expected only the two fully-valid rows");
        assert_eq!(rows[0].timestamp_ms, 1000);
        assert_eq!(rows[1].timestamp_ms, 3000);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn telemetry_snapshot_for_tick_wraps_around_csv_rows() {
        let rows = vec![
            corinth_canal::TelemetrySnapshot {
                timestamp_ms: 111,
                gpu_temp_c: 10.0,
                gpu_power_w: 100.0,
                cpu_tctl_c: 20.0,
                cpu_package_power_w: 200.0,
                heartbeat_signal: 0.0,
                heartbeat_enabled: false,
            },
            corinth_canal::TelemetrySnapshot {
                timestamp_ms: 222,
                gpu_temp_c: 30.0,
                gpu_power_w: 300.0,
                cpu_tctl_c: 40.0,
                cpu_package_power_w: 400.0,
                heartbeat_signal: 0.0,
                heartbeat_enabled: false,
            },
        ];
        let resolved = ResolvedTelemetry {
            source: TelemetrySource::Csv,
            source_label: "csv_re4".to_string(),
            csv_path: Some(PathBuf::from("/tmp/telemetry.csv")),
            rows: Some(rows),
        };

        // tick=0 uses row[0], tick=3 wraps back to row[1] (3 % 2 == 1).
        let snap0 = telemetry_snapshot_for_tick(0, &resolved);
        let snap3 = telemetry_snapshot_for_tick(3, &resolved);

        assert!((snap0.gpu_temp_c - 10.0).abs() < 1e-6);
        assert!((snap3.gpu_temp_c - 30.0).abs() < 1e-6);
        // timestamps are rewritten to (tick + 1) for 1-to-1 join with tick txt.
        assert_eq!(snap0.timestamp_ms, 1);
        assert_eq!(snap3.timestamp_ms, 4);
    }

    #[test]
    fn telemetry_snapshot_for_tick_uses_synthetic_on_fallback() {
        let resolved = ResolvedTelemetry {
            source: TelemetrySource::Synthetic,
            source_label: "synthetic_fallback".to_string(),
            csv_path: Some(PathBuf::from("/nonexistent/telemetry.csv")),
            rows: None,
        };
        let snap = telemetry_snapshot_for_tick(5, &resolved);
        // Synthetic path writes its own timestamp, then we overwrite to tick+1.
        assert_eq!(snap.timestamp_ms, 6);
        // And the synthetic sinusoid produces non-zero, finite values.
        assert!(snap.gpu_temp_c.is_finite() && snap.gpu_temp_c > 0.0);
    }

    #[test]
    fn csv_source_label_maps_telemetry_stem_to_csv_re4() {
        assert_eq!(csv_source_label(Path::new("telemetry.csv")), "csv_re4");
        assert_eq!(
            csv_source_label(Path::new("/tmp/cyberpunk2077_combat.csv")),
            "csv_cyberpunk2077_combat"
        );
    }

    #[test]
    fn prompt_embedding_provider_defaults_to_ollama() {
        assert_eq!(
            resolve_prompt_embedding_provider(None),
            PromptEmbeddingProvider::Ollama
        );
    }

    #[test]
    fn prompt_embedding_provider_accepts_ollama_case_insensitively() {
        assert_eq!(
            resolve_prompt_embedding_provider(Some("  OLLAMA  ")),
            PromptEmbeddingProvider::Ollama
        );
    }

    #[test]
    fn prompt_embedding_provider_rejects_legacy_llama_cpp() {
        assert_eq!(
            resolve_prompt_embedding_provider(Some("llama_cpp")),
            PromptEmbeddingProvider::SyntheticFallback
        );
    }
}
