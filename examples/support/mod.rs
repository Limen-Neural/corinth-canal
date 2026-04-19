//! Shared helper functions for the example binaries.
#![allow(dead_code)]

use corinth_canal::{
    HeartbeatConfig, HeartbeatInjector, ModelFamily, SaaqUpdateRule, EMBEDDING_DIM,
    model::ModelConfig,
    moe::RoutingMode,
    moe::OlmoeRouter,
    projector::ProjectionMode,
};
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_MATH_PROMPT_TOKEN_IDS: [usize; 9] =
    [402, 11492, 286, 257, 4568, 318, 12056, 4202, 13];
pub const DEFAULT_MATH_PROMPT_TEXT: &str = "The derivative of a constant is mathematically zero.";

#[derive(Debug, Clone)]
pub struct ValidationModelSpec {
    pub slug: String,
    pub family: Option<ModelFamily>,
    pub path: String,
}

pub fn gguf_checkpoint_path_or_default() -> String {
    std::env::var("GGUF_CHECKPOINT_PATH").unwrap_or_default()
}

pub fn required_gguf_checkpoint_path() -> Result<String, Box<dyn std::error::Error>> {
    let model_path = gguf_checkpoint_path_or_default();
    if model_path.trim().is_empty() {
        return Err(Error::other("GGUF_CHECKPOINT_PATH must point to a GGUF checkpoint").into());
    }
    Ok(model_path)
}

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
            .and_then(|metadata| metadata.preferred_gpu_synapse_tensor_name.clone())
            .unwrap_or_default(),
        num_experts: probe.as_ref().map(|metadata| metadata.num_experts).unwrap_or(8),
        top_k_experts: probe
            .as_ref()
            .map(|metadata| metadata.expert_used_count.max(1))
            .unwrap_or(1),
        routing_mode: RoutingMode::SpikingSim,
        snn_steps,
        projection_mode: ProjectionMode::SpikingTernary,
        heartbeat: heartbeat_config_from_env(),
    }
}

pub fn mean_pool_prompt_embeddings(
    model: &mut corinth_canal::model::Model,
    token_ids: &[usize],
) -> corinth_canal::Result<Vec<f32>> {
    let mut pooled = vec![0.0f32; EMBEDDING_DIM];

    for &token in token_ids {
        let emb = model.extract_token_embedding(token)?;
        for (dst, src) in pooled.iter_mut().zip(emb.iter()) {
            *dst += *src;
        }
    }

    normalize_embedding(&mut pooled);
    Ok(pooled)
}

pub fn prompt_profile_slug() -> String {
    std::env::var("PROMPT_PROFILE")
        .unwrap_or_else(|_| "math_logic".into())
        .to_ascii_lowercase()
}

pub fn prompt_text_for_profile(profile: &str) -> &'static str {
    match profile {
        "math_logic" | "math" => DEFAULT_MATH_PROMPT_TEXT,
        _ => DEFAULT_MATH_PROMPT_TEXT,
    }
}

pub fn model_family_override_from_env() -> Option<ModelFamily> {
    let value = std::env::var("MODEL_FAMILY").ok()?;
    match value.to_ascii_lowercase().as_str() {
        "olmoe" => Some(ModelFamily::Olmoe),
        "qwen3moe" | "qwen3_moe" | "qwen" => Some(ModelFamily::Qwen3Moe),
        "gemma4" | "gemma_4" | "gemma" => Some(ModelFamily::Gemma4),
        "deepseek2" | "deepseek_v2" | "deepseek" => Some(ModelFamily::DeepSeek2),
        "llama" | "llama_moe" | "llama3_moe" => Some(ModelFamily::LlamaMoe),
        _ => None,
    }
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

pub fn prompt_embedding_for_validation(
    model_path: &str,
    prompt_text: &str,
    target_dim: usize,
) -> Result<(Vec<f32>, String), Box<dyn std::error::Error>> {
    match pooled_prompt_embedding_from_llama_cpp(model_path, prompt_text, target_dim) {
        Ok(embedding) => Ok((embedding, "llama_cpp_mean_pool".into())),
        Err(error) => {
            eprintln!(
                "llama.cpp prompt embedding unavailable for '{}': {}. Falling back to deterministic text hash embedding.",
                model_path, error
            );
            Ok((
                synthetic_text_embedding(prompt_text, target_dim),
                "text_hash_fallback".into(),
            ))
        }
    }
}

pub fn pooled_prompt_embedding_from_llama_cpp(
    model_path: &str,
    prompt_text: &str,
    target_dim: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let binary = llama_embedding_binary()?;
    let output = Command::new(binary)
        .arg("-m")
        .arg(model_path)
        .arg("-p")
        .arg(prompt_text)
        .arg("--pooling")
        .arg("mean")
        .arg("--no-warmup")
        .arg("--embd-output-format")
        .arg("json")
        .arg("-n")
        .arg("0")
        .arg("-ngl")
        .arg("0")
        .output()?;

    if !output.status.success() {
        return Err(Error::other(format!(
            "llama-embedding failed for '{model_path}': {}",
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }

    let stdout = String::from_utf8(output.stdout)?;
    let mut embedding = parse_llama_embedding_payload(&stdout)?;
    if embedding.len() != target_dim {
        embedding = resample_embedding(&embedding, target_dim);
    }
    normalize_embedding(&mut embedding);
    Ok(embedding)
}

pub fn discover_validation_models() -> Vec<ValidationModelSpec> {
    if let Ok(path) = std::env::var("GGUF_CHECKPOINT_PATH") {
        if !path.trim().is_empty() {
            let family = OlmoeRouter::probe_model(&path, None).ok().map(|metadata| metadata.family);
            return vec![ValidationModelSpec {
                slug: slug_from_path(&path),
                family,
                path,
            }];
        }
    }

    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let root = PathBuf::from(home).join("Downloads").join("SNN_Quantization");
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
            PathBuf::from("models/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q6_K_L.gguf"),
        ),
        (
            "llama_3_2_dark_champion_q5_k_m",
            Some(ModelFamily::LlamaMoe),
            PathBuf::from("models/Llama-3.2-8X3B-MOE-Dark-Champion-GGUF/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-q5_k_m.gguf"),
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
            })
        })
        .collect()
}

pub fn ticks_from_env(default_ticks: usize) -> usize {
    env_usize("TICKS", default_ticks)
}

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

pub fn heartbeat_gain(signal: f32) -> f32 {
    (1.0 + signal * 0.28).max(0.15)
}

pub fn heartbeat_injector_from_env() -> HeartbeatInjector {
    HeartbeatInjector::new(heartbeat_config_from_env())
}

fn llama_embedding_binary() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Ok(path) = std::env::var("LLAMA_EMBEDDING_BIN") {
        let binary = PathBuf::from(path);
        if binary.exists() {
            return Ok(binary);
        }
    }

    let local = PathBuf::from("/home/raulmc/llama.cpp/build/bin/llama-embedding");
    if local.exists() {
        return Ok(local);
    }

    Err(Error::other("LLAMA_EMBEDDING_BIN must point to llama.cpp's llama-embedding binary").into())
}

fn parse_llama_embedding_payload(
    stdout: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    #[derive(serde::Deserialize)]
    struct JsonEmbeddingRow {
        embedding: Vec<f32>,
    }

    #[derive(serde::Deserialize)]
    struct JsonEmbeddingList {
        data: Vec<JsonEmbeddingRow>,
    }

    if let Ok(payload) = serde_json::from_str::<JsonEmbeddingList>(stdout) {
        if let Some(row) = payload.data.into_iter().next() {
            return Ok(row.embedding);
        }
    }

    if let Ok(payload) = serde_json::from_str::<Vec<Vec<f32>>>(stdout) {
        if let Some(row) = payload.into_iter().next() {
            return Ok(row);
        }
    }

    Err(Error::other("llama-embedding output did not contain a JSON embedding payload").into())
}

fn slug_from_path(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("gguf_model")
        .replace(['.', '-', ' '], "_")
        .to_ascii_lowercase()
}

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
            continue;
        }
        let t = source - lo as f32;
        out.push(input[lo] * (1.0 - t) + input[hi] * t);
    }
    out
}

fn normalize_embedding(values: &mut [f32]) {
    let l2_norm = values.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if l2_norm > 1e-8 {
        for value in values {
            *value /= l2_norm;
        }
    }
}

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

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

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

fn env_flag(key: &str, default_value: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default_value)
}
