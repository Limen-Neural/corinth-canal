//! Single source of env-driven runtime config for example binaries.
//!
//! The runtime crate (`src/`) never reads environment variables for paths.
//! Every machine-local default lives here. Each example binary should start
//! with:
//!
//! ```ignore
//! let _ = dotenvy::from_filename(".env.local");
//! let cfg = support::RunConfig::from_env();
//! ```
//!
//! Fields map 1:1 to the entries documented in `.env.example`.

use std::path::{Path, PathBuf};

use corinth_canal::{ModelFamily, SaaqUpdateRule, moe::RoutingMode};
use serde::Deserialize;

use super::{
    ResolvedTelemetry, SafetensorsModelEntry, ValidationModelSpec, cloud_execution_guard,
    cloud_lineup_path_from_env, discover_validation_models, env_flag, heartbeat_modes_for_matrix,
    load_cloud_lineup, load_safetensors_lineup, model_family_override_from_env, parse_family_slug,
    parse_routing_mode, prompt_profile_slug, prompt_text_for_profile, repeat_count_from_env,
    resolve_telemetry_source, routing_mode_override_from_env, saaq_update_rule_from_env,
    safetensors_lineup_path_from_env, ticks_from_env,
};

/// Default output root for per-run artifacts when `VALIDATION_OUTPUT_ROOT`
/// is unset. Repo-relative on purpose so a fresh clone never writes into a
/// machine-specific consumer directory.
pub const DEFAULT_OUTPUT_ROOT: &str = "artifacts";

/// Default tick count for `saaq_latent_calibration` when `TICKS` is unset.
pub const DEFAULT_TICKS: usize = 512;

/// Aggregated env-driven configuration for an example binary run.
///
/// Every field is populated by `RunConfig::from_env()` in one pass. Binaries
/// should not read `std::env` directly. Each example binary reads only a
/// subset of these fields, so the struct is tagged `#[allow(dead_code)]`
/// to silence per-binary `field never read` warnings.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub prompt_profile: String,
    pub prompt_text: &'static str,
    pub ticks: usize,
    pub repeat_count: usize,
    pub heartbeat_matrix: Vec<bool>,
    pub telemetry: ResolvedTelemetry,
    pub output_root: PathBuf,
    pub model_family_override: Option<ModelFamily>,
    pub saaq_rule: SaaqUpdateRule,
    pub validation_models: Vec<ValidationModelSpec>,
    pub gguf_checkpoint_path: String,
    pub routing_mode_override: Option<RoutingMode>,
    /// Free-form run tag from `RUN_TAG`. Empty / unset maps to `None` so
    /// callers can just do `if let Some(tag) = cfg.run_tag { ... }`.
    pub run_tag: Option<String>,
    /// When `true` and `repeat_count >= 2`, the calibration runner asserts
    /// byte-equality of `latent_telemetry.csv` across repeats per
    /// `(model_slug, telemetry_source, heartbeat_slug, saaq_rule)` group.
    pub strict_repeat_check: bool,
}

impl RunConfig {
    /// Build a `RunConfig` by reading every supported environment variable.
    ///
    /// Call `dotenvy::from_filename(".env.local").ok();` before this if you
    /// want `.env.local` overrides applied.
    pub fn from_env() -> Self {
        let prompt_profile = prompt_profile_slug();
        let prompt_text = prompt_text_for_profile(&prompt_profile);
        let gguf_checkpoint_path = std::env::var("GGUF_CHECKPOINT_PATH").unwrap_or_default();
        validate_optional_lineups_from_env();
        // Local binding only — used to pick the lineup TOML below. The
        // resolved path is intentionally not stamped onto `RunConfig` yet;
        // when campaign provenance is added back to `ValidationManifest`,
        // re-introduce both fields together in one focused commit.
        let lineup_config_path = lineup_config_path_from_env();
        let validation_models =
            resolve_validation_models(lineup_config_path.as_deref(), &gguf_checkpoint_path);
        Self {
            prompt_profile: prompt_profile.clone(),
            prompt_text,
            ticks: ticks_from_env(DEFAULT_TICKS),
            repeat_count: repeat_count_from_env(),
            heartbeat_matrix: heartbeat_modes_for_matrix(),
            telemetry: resolve_telemetry_source(),
            output_root: output_root_from_env(),
            model_family_override: model_family_override_from_env(),
            saaq_rule: saaq_update_rule_from_env(),
            validation_models,
            gguf_checkpoint_path,
            routing_mode_override: routing_mode_override_from_env(),
            run_tag: run_tag_from_env(),
            strict_repeat_check: strict_repeat_check_from_env(),
        }
    }
}

fn validate_optional_lineups_from_env() {
    if let Some(path) = cloud_lineup_path_from_env() {
        let entries = load_cloud_lineup(&path).unwrap_or_else(|err| {
            panic!(
                "CLOUD_LINEUP_CONFIG={} could not be loaded: {err}",
                path.display()
            )
        });
        for entry in &entries {
            cloud_execution_guard(entry).unwrap_or_else(|err| {
                panic!(
                    "CLOUD_LINEUP_CONFIG={} failed validation for slug={}: {err}",
                    path.display(),
                    entry.slug
                )
            });
        }
    }

    if let Some(path) = safetensors_lineup_path_from_env() {
        let entries = load_safetensors_lineup(&path).unwrap_or_else(|err| {
            panic!(
                "SAFETENSORS_LINEUP_CONFIG={} could not be loaded: {err}",
                path.display()
            )
        });
        validate_safetensors_lineup_entries(&path, &entries);
    }
}

fn validate_safetensors_lineup_entries(path: &Path, entries: &[SafetensorsModelEntry]) {
    if entries.is_empty() {
        panic!(
            "SAFETENSORS_LINEUP_CONFIG={} produced no usable entries. \
             Placeholder-only or missing-path lineups are not valid runtime config.",
            path.display()
        );
    }
    for entry in entries {
        if entry.slug.trim().is_empty() {
            panic!(
                "SAFETENSORS_LINEUP_CONFIG={} contains an empty slug for path={}",
                path.display(),
                entry.path.display()
            );
        }
        if !entry.target.eq_ignore_ascii_case("local") {
            panic!(
                "SAFETENSORS_LINEUP_CONFIG={} has invalid target={} for slug={}",
                path.display(),
                entry.target,
                entry.slug
            );
        }
        if !entry.path.exists() {
            panic!(
                "SAFETENSORS_LINEUP_CONFIG={} resolved missing path={} for slug={}",
                path.display(),
                entry.path.display(),
                entry.slug
            );
        }
        if let Some(family) = entry.family {
            let _ = family.slug();
        }
    }
}

/// Resolve the validation-model list with the documented precedence:
///
///   1. `LINEUP_CONFIG` file (hard error if set but unparseable).
///   2. `GGUF_CHECKPOINT_PATH` (single-model override via the legacy path).
///   3. Machine-local autodiscovery under `$HOME/Downloads/SNN_Quantization`.
fn resolve_validation_models(
    lineup_path: Option<&Path>,
    gguf_checkpoint_path: &str,
) -> Vec<ValidationModelSpec> {
    if let Some(path) = lineup_path {
        match load_lineup_file(path) {
            Ok(models) => return models,
            Err(err) => {
                let path_str = path.display().to_string();
                let hint = if path_str.contains("/absolute/path/to/") {
                    "\n\nHINT: The path appears to be a placeholder from .env.example or a config template.\n      Please update LINEUP_CONFIG in .env.local with a real path."
                } else {
                    ""
                };
                // Hard-fail with a loud message so a typo / missing path is
                // never silently papered over by autodiscovery.
                panic!("LINEUP_CONFIG={path_str} could not be loaded: {err}{hint}");
            }
        }
    }

    // Legacy single-model override or autodiscovery — let the existing
    // helper keep its current contract.
    let _ = gguf_checkpoint_path; // discover_validation_models reads it directly
    discover_validation_models()
}

#[derive(Debug, Deserialize)]
struct RawLineup {
    #[serde(default)]
    model: Vec<RawLineupModel>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLineupModel {
    slug: String,
    family: String,
    path: String,
    #[serde(default)]
    routing_mode: Option<String>,
}

/// Parse the lineup TOML and convert each entry to a `ValidationModelSpec`.
/// Missing files are skipped with a warning (same non-fatal behavior as
/// `discover_validation_models`); unknown family / routing-mode slugs are
/// reported to stderr but do not abort the run.
fn load_lineup_file(path: &Path) -> Result<Vec<ValidationModelSpec>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let parsed: RawLineup =
        toml::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))?;

    let mut out = Vec::with_capacity(parsed.model.len());
    for entry in parsed.model {
        let RawLineupModel {
            slug,
            family,
            path: gguf_path,
            routing_mode,
        } = entry;
        let trimmed_path = gguf_path.trim();
        if trimmed_path.is_empty() {
            eprintln!("lineup_config: skipping entry slug={slug}: empty path",);
            continue;
        }
        if !Path::new(trimmed_path).exists() {
            let hint = if trimmed_path.contains("/absolute/path/to/") {
                " (appears to be a placeholder path)"
            } else {
                ""
            };
            eprintln!(
                "lineup_config: skipping entry slug={slug} path={trimmed_path}: file not found{hint}",
            );
            continue;
        }
        let parsed_family = parse_family_slug(&family);
        if parsed_family.is_none() {
            eprintln!(
                "lineup_config: unknown family '{family}' for slug={slug}; leaving family inference to probe",
            );
        }
        let parsed_routing = match routing_mode.as_deref() {
            Some(value) => {
                let resolved = parse_routing_mode(value);
                if resolved.is_none() {
                    eprintln!(
                        "lineup_config: unknown routing_mode '{value}' for slug={slug}; using ModelConfig default",
                    );
                }
                resolved
            }
            None => None,
        };

        out.push(ValidationModelSpec {
            slug,
            family: parsed_family,
            path: trimmed_path.to_owned(),
            routing_mode: parsed_routing,
        });
    }

    Ok(out)
}

/// Parse `LINEUP_CONFIG`. Empty / unset => `None`.
pub fn lineup_config_path_from_env() -> Option<PathBuf> {
    std::env::var("LINEUP_CONFIG")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

/// Parse `RUN_TAG`. Empty / unset => `None`. Whitespace-only values are
/// normalized to `None`.
pub fn run_tag_from_env() -> Option<String> {
    std::env::var("RUN_TAG")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// Parse `STRICT_REPEAT_CHECK`. Default `false` so existing workflows keep
/// their current behavior when the env var is unset.
pub fn strict_repeat_check_from_env() -> bool {
    env_flag("STRICT_REPEAT_CHECK", false)
}

/// Resolve `VALIDATION_OUTPUT_ROOT`, falling back to the repo-relative
/// default `./artifacts`.
pub fn output_root_from_env() -> PathBuf {
    if let Ok(value) = std::env::var("VALIDATION_OUTPUT_ROOT") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from(DEFAULT_OUTPUT_ROOT)
}

#[cfg(test)]
mod tests {
    use super::validate_safetensors_lineup_entries;
    use crate::support::SafetensorsModelEntry;
    use corinth_canal::ModelFamily;
    use std::path::{Path, PathBuf};

    #[test]
    #[should_panic(expected = "produced no usable entries")]
    fn safetensors_validation_rejects_empty_lineup() {
        validate_safetensors_lineup_entries(Path::new("configs/template.toml"), &[]);
    }

    #[test]
    fn safetensors_validation_accepts_existing_entry() {
        let existing_file = std::env::temp_dir().join(format!(
            "st_validation_{}.safetensors",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&existing_file, b"test").unwrap();

        let entries = vec![SafetensorsModelEntry {
            slug: "test_st_model".into(),
            family: Some(ModelFamily::Olmoe),
            path: existing_file.clone(),
            target: "local".into(),
        }];

        validate_safetensors_lineup_entries(Path::new("configs/runtime.toml"), &entries);

        let _ = std::fs::remove_file(existing_file);
    }
}
