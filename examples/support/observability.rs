#![allow(dead_code)]

use std::borrow::Cow;
use std::path::Path;
use std::process::Command;
use std::sync::Once;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use sentry::ClientInitGuard;
use serde_json::json;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();
const REPO_NAME: &str = "corinth-canal";

#[derive(Debug, Clone, Copy, Default)]
pub struct SafeDiagnosticData<'a> {
    pub model_slug: Option<&'a str>,
    pub telemetry_source: Option<&'a str>,
    pub heartbeat_enabled: Option<bool>,
    pub validation_status: Option<&'a str>,
    pub error_category: Option<&'a str>,
}

impl<'a> SafeDiagnosticData<'a> {
    pub fn with_model_slug(mut self, model_slug: &'a str) -> Self {
        self.model_slug = Some(model_slug);
        self
    }

    pub fn with_telemetry_source(mut self, telemetry_source: &'a str) -> Self {
        self.telemetry_source = Some(telemetry_source);
        self
    }

    pub fn with_heartbeat_enabled(mut self, heartbeat_enabled: bool) -> Self {
        self.heartbeat_enabled = Some(heartbeat_enabled);
        self
    }

    pub fn with_validation_status(mut self, validation_status: &'a str) -> Self {
        self.validation_status = Some(validation_status);
        self
    }

    pub fn with_error_category(mut self, error_category: &'a str) -> Self {
        self.error_category = Some(error_category);
        self
    }
}

pub struct CommandObserver {
    command: &'static str,
    run_id: String,
    git_sha: String,
    started: Instant,
}

pub trait ErrorReport {
    fn as_dyn_error(&self) -> &(dyn std::error::Error + 'static);
}

pub fn init_tracing() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        let builder = tracing_subscriber::fmt().with_env_filter(filter);

        if std::env::var("AGENTOS_JSON_TRACING").as_deref() == Ok("1") {
            builder.json().init();
        } else {
            builder.init();
        }
    });
}

pub fn start_command(command: &'static str) -> CommandObserver {
    init_tracing();
    let observer = CommandObserver {
        command,
        run_id: run_id(),
        git_sha: git_sha(),
        started: Instant::now(),
    };
    annotate_scope(
        observer.command,
        &observer.run_id,
        &observer.git_sha,
        SafeDiagnosticData::default(),
    );
    tracing::info!(
        event = "command_start",
        repo = REPO_NAME,
        command = observer.command,
        run_id = %observer.run_id,
        git_sha = %observer.git_sha,
        "command_start"
    );
    observer
}

pub fn init_sentry(command: &'static str) -> Option<ClientInitGuard> {
    let dsn = std::env::var("SENTRY_DSN")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())?;
    let git_sha = git_sha();
    let release = resolve_sentry_release(&git_sha);
    let environment = std::env::var("SENTRY_ENVIRONMENT")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "local".to_owned());
    let parsed_dsn = match dsn.parse() {
        Ok(dsn) => dsn,
        Err(error) => {
            eprintln!("Sentry disabled: invalid SENTRY_DSN ({error})");
            return None;
        }
    };

    let guard = sentry::init(sentry::ClientOptions {
        dsn: Some(parsed_dsn),
        release: Some(Cow::Owned(release)),
        environment: Some(Cow::Owned(environment)),
        sample_rate: 1.0,
        traces_sample_rate: 0.0,
        default_integrations: true,
        ..Default::default()
    });

    annotate_scope(command, &run_id(), &git_sha, SafeDiagnosticData::default());

    Some(guard)
}

pub fn capture_top_level_error(command: &'static str, error: &(dyn std::error::Error + 'static)) {
    capture_scoped_error(command, &run_id(), SafeDiagnosticData::default(), error);
}

pub fn annotate_scope(
    command: &'static str,
    run_id: &str,
    git_sha: &str,
    data: SafeDiagnosticData<'_>,
) {
    if sentry::Hub::with_active(|hub| hub.client().is_none()) {
        return;
    }

    sentry::configure_scope(|scope| {
        apply_scope(scope, command, run_id, git_sha, data);
    });
}

pub fn capture_scoped_error(
    command: &'static str,
    run_id: &str,
    data: SafeDiagnosticData<'_>,
    error: &(dyn std::error::Error + 'static),
) {
    if sentry::Hub::with_active(|hub| hub.client().is_none()) {
        return;
    }

    let git_sha = git_sha();
    sentry::with_scope(
        |scope| {
            apply_scope(scope, command, run_id, &git_sha, data);
        },
        || {
            sentry::capture_error(error);
        },
    );
}

pub fn checkpoint_slug(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }

    Some(
        Path::new(trimmed)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("gguf_model")
            .replace(['.', '-', ' '], "_")
            .to_ascii_lowercase(),
    )
}

pub fn run_id() -> String {
    std::env::var("AGENTOS_RUN_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            let millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_millis())
                .unwrap_or(0);
            format!("corinth-canal-{millis}")
        })
}

/// Resolve the git SHA stamped into observability events.
///
/// Precedence:
///   1. `AGENTOS_GIT_SHA` (set by managed CI / AgentOS).
///   2. `git rev-parse --short HEAD` invoked from the current working
///      directory. This matches the fallback already used by
///      `scripts/observability/newrelic_event.sh` and
///      `scripts/observability/sentry_release.sh`, so Rust traces and
///      shell-emitted releases / events agree on the SHA for the same
///      checkout (PR #30 review consistency requirement).
///   3. Literal `"unknown"` when neither source resolves (e.g. running
///      from outside a git checkout with the env var unset).
pub fn git_sha() -> String {
    if let Some(value) = std::env::var("AGENTOS_GIT_SHA")
        .ok()
        .filter(|value| !value.trim().is_empty())
    {
        return value;
    }
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        if output.status.success() {
            let sha = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if !sha.is_empty() {
                return sha;
            }
        }
    }
    "unknown".to_owned()
}

fn resolve_sentry_release(git_sha: &str) -> String {
    if let Some(release) = std::env::var("SENTRY_RELEASE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        return release;
    }

    if let Some(release) = sentry::release_name!() {
        let release = release.into_owned();
        if !release.trim().is_empty() {
            return release;
        }
    }

    format!("corinth-canal@{git_sha}")
}

impl CommandObserver {
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn git_sha(&self) -> &str {
        &self.git_sha
    }

    pub fn annotate(&self, data: SafeDiagnosticData<'_>) {
        annotate_scope(self.command, &self.run_id, &self.git_sha, data);
    }

    pub fn finish<T, E>(&self, result: &Result<T, E>, data: SafeDiagnosticData<'_>)
    where
        E: ErrorReport + ToString,
    {
        let error_message = result.as_ref().err().map(|error| error.to_string());
        let resolved_status = data.validation_status.unwrap_or(if result.is_ok() {
            "completed"
        } else {
            "failed"
        });
        let resolved_category = data.error_category.unwrap_or(error_category(
            Some(resolved_status),
            error_message.as_deref(),
        ));
        let enriched = data
            .with_validation_status(resolved_status)
            .with_error_category(resolved_category);

        annotate_scope(self.command, &self.run_id, &self.git_sha, enriched);
        tracing::info!(
            event = "command_finish",
            repo = REPO_NAME,
            command = self.command,
            run_id = %self.run_id,
            git_sha = %self.git_sha,
            latency_ms = self.started.elapsed().as_millis() as u64,
            success = result.is_ok(),
            error_category = resolved_category,
            validation_status = resolved_status,
            "command_finish"
        );

        if let Err(error) = result.as_ref() {
            capture_scoped_error(self.command, &self.run_id, enriched, error.as_dyn_error());
        }
    }
}

impl ErrorReport for Box<dyn std::error::Error> {
    fn as_dyn_error(&self) -> &(dyn std::error::Error + 'static) {
        self.as_ref()
    }
}

impl ErrorReport for corinth_canal::HybridError {
    fn as_dyn_error(&self) -> &(dyn std::error::Error + 'static) {
        self
    }
}

fn apply_scope(
    scope: &mut sentry::Scope,
    command: &'static str,
    run_id: &str,
    git_sha: &str,
    data: SafeDiagnosticData<'_>,
) {
    scope.set_tag("repo", REPO_NAME);
    scope.set_tag("command", command);
    scope.set_tag("git_sha", git_sha.to_owned());
    if let Some(model_slug) = data.model_slug {
        scope.set_tag("model_slug", model_slug);
    }
    if let Some(telemetry_source) = data.telemetry_source {
        scope.set_tag("telemetry_source", telemetry_source);
    }
    if let Some(heartbeat_enabled) = data.heartbeat_enabled {
        scope.set_tag("heartbeat_enabled", heartbeat_enabled.to_string());
    }
    if let Some(validation_status) = data.validation_status {
        scope.set_tag("validation_status", validation_status);
    }
    if let Some(error_category) = data.error_category {
        scope.set_tag("error_category", error_category);
    }
    scope.set_extra("run_id", json!(run_id));
}

pub fn error_category(status: Option<&str>, error: Option<&str>) -> &'static str {
    match status.unwrap_or_default() {
        "completed" => "none",
        "prompt_embedding_failed" => "config_error",
        // Model::new() and router-load failures both stem from checkpoint
        // / runtime configuration problems (missing GGUF metadata, bad
        // family override, unreachable path). Map them deterministically
        // here so observability dashboards never see them fall through to
        // the substring heuristic below.
        "model_setup_failed" => "config_error",
        "router_load_failed" => "config_error",
        "gpu_setup_failed" => "gpu_error",
        "tick_failed" => "experiment_error",
        _ => {
            let message = error.unwrap_or_default().to_ascii_lowercase();
            if message.contains("strict_repeat_check") {
                "experiment_error"
            } else if message.contains("gpu") || message.contains("cuda") {
                "gpu_error"
            } else if message.contains("checkpoint")
                || message.contains("config")
                || message.contains("no gguf")
            {
                "config_error"
            } else if error.is_some() {
                "unknown_error"
            } else {
                "none"
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn clear_env(key: &str) {
        unsafe {
            std::env::remove_var(key);
        }
    }

    fn set_env(key: &str, value: &str) {
        unsafe {
            std::env::set_var(key, value);
        }
    }

    #[test]
    fn run_id_uses_agentos_run_id_when_set() {
        let _guard = env_lock().lock().unwrap();
        set_env("AGENTOS_RUN_ID", "run-123");
        assert_eq!(run_id(), "run-123");
        clear_env("AGENTOS_RUN_ID");
    }

    #[test]
    fn run_id_falls_back_to_corinth_canal_millis() {
        let _guard = env_lock().lock().unwrap();
        clear_env("AGENTOS_RUN_ID");
        let value = run_id();
        assert!(value.starts_with("corinth-canal-"));
        assert!(value["corinth-canal-".len()..].parse::<u128>().is_ok());
    }

    #[test]
    fn git_sha_uses_agentos_git_sha_when_set() {
        let _guard = env_lock().lock().unwrap();
        set_env("AGENTOS_GIT_SHA", "deadbee");
        assert_eq!(git_sha(), "deadbee");
        clear_env("AGENTOS_GIT_SHA");
    }

    #[test]
    fn error_category_maps_known_statuses() {
        assert_eq!(error_category(Some("completed"), None), "none");
        assert_eq!(
            error_category(Some("model_setup_failed"), Some("boom")),
            "config_error"
        );
        assert_eq!(
            error_category(Some("router_load_failed"), Some("boom")),
            "config_error"
        );
        assert_eq!(
            error_category(Some("gpu_setup_failed"), Some("boom")),
            "gpu_error"
        );
        assert_eq!(
            error_category(Some("tick_failed"), Some("boom")),
            "experiment_error"
        );
    }

    #[test]
    fn error_category_uses_substring_fallbacks() {
        assert_eq!(
            error_category(None, Some("strict_repeat_check mismatch")),
            "experiment_error"
        );
        assert_eq!(error_category(None, Some("GPU device failed")), "gpu_error");
        assert_eq!(
            error_category(None, Some("cuda launch failed")),
            "gpu_error"
        );
        assert_eq!(
            error_category(None, Some("checkpoint metadata missing")),
            "config_error"
        );
        assert_eq!(
            error_category(None, Some("config parse failed")),
            "config_error"
        );
        assert_eq!(
            error_category(None, Some("mystery failure")),
            "unknown_error"
        );
    }

    #[test]
    fn unset_sentry_dsn_disables_sentry_cleanly() {
        let _guard = env_lock().lock().unwrap();
        clear_env("SENTRY_DSN");
        clear_env("SENTRY_RELEASE");
        clear_env("SENTRY_ENVIRONMENT");
        assert!(init_sentry("test_command").is_none());
    }

    #[test]
    fn empty_sentry_dsn_disables_sentry_cleanly() {
        let _guard = env_lock().lock().unwrap();
        set_env("SENTRY_DSN", "   ");
        clear_env("SENTRY_RELEASE");
        clear_env("SENTRY_ENVIRONMENT");
        assert!(init_sentry("test_command").is_none());
        clear_env("SENTRY_DSN");
    }
}
