# AGENTS.md

## Project identity

`corinth-canal` is the single-crate reference implementation for the `rmems`
SNN-logic quantization line.

The core pipeline is:

TelemetrySnapshot
-> TelemetryEncoder / GPU snapshot projection
-> ternary spikes
-> SignedSplitBankBridge or GPU input_spikes
-> SparseGifHiddenLayer / GPU GIF temporal loop
-> Projector
-> OlmoeRouter
-> routing telemetry / SAAQ latent calibration

This repository is intentionally not split into modular crates yet. Do not
extract modules into separate crates unless explicitly asked. Proven components
graduate later according to `docs/PROMOTION_RULES.md` and
`docs/MODULE_STATUS.md`.

## Agent and executor tooling

Primary executor for this repository:

- Codex is the primary executor for repository tasks and follow-up
  implementation work.

Memory-agent backbone:

- Hermes Agents and GitHub Copilot are approved at a high level as part of the
  memory-agent backbone for planning, recall, review support, and follow-up
  execution assistance.

Approved fallback executors:

- Opencode — backup executor path, including free-API usage scenarios.
- Kilo — fallback executor for repository tasks.
- Cline — fallback executor for repository tasks.
- Gemini CLI — occasional fallback for large-context tasks.

Experimental tooling:

- Goose Agent — experimental only. Do not treat it as the default execution
  path.
- OpenClaw — experimental only. Do not treat it as the default execution path.
- Blocks — experimental only. Blocks uses the maintainer's ChatGPT Pro
  subscription and should be treated as an optional, non-default path.

Optional third-party integrations and evaluated tooling:

- Google APIs
- NVIDIA NIM
- OpenRouter
- Grok APIs
- CodeHawk — optional evaluated tooling; currently available via a 30-day free
  trial.
- IBM bob — potential evaluated tooling; needs verification before being
  documented as enabled.

These integrations and tool references are optional and should only be
referenced at a high level in repository docs and examples.

Safety and documentation rules for executor tooling:

- Do not commit or document secrets, tokens, DSNs, API keys, private telemetry,
  or local absolute paths.
- Do not assume optional third-party APIs are configured.
- Local/offline behavior must remain safe when optional external integrations
  are unset.
- Documentation changes about agent tooling should stay high level and should
  not publish machine-specific setup details unless explicitly requested.
- Keep executor/tooling documentation limited to markdown and closely related
  repo docs unless a task explicitly asks for implementation work.

## Non-negotiable rules

- Keep this as a single Rust crate unless the task explicitly says otherwise.
- Do not delete or bypass the SNN/GIF/SAAQ routing logic.
- Do not replace real GGUF-backed routing with stubs except in tests or explicit
  fallback paths.
- Do not introduce machine-local absolute paths such as `/home/...` into `src/`.
- Do not hardcode checkpoint paths, telemetry CSV paths, CUDA paths, or artifact
  output paths.
- Do not add new dependencies unless they are clearly justified.
- Preserve CPU-only buildability.
- Preserve CUDA/GPU behavior when touching GPU code.
- Keep diffs small and reviewable.
- Add or update tests when behavior changes.
- Do not change CSV schemas unless explicitly instructed.
- Do not silently change public APIs exported from `src/lib.rs`.
- Do not touch generated artifacts unless the task is specifically about run
  manifests, validation outputs, or known-good run logs.

## Repository structure

Important paths:

- `src/model/core.rs`
  - Runtime orchestration, config validation, forward paths.
- `src/model/temporal.rs`
  - GPU temporal loop:
    `prepare_gpu_temporal`, `tick_gpu_temporal`, `forward_gpu_temporal`.
- `src/model/telemetry_io.rs`
  - Routing telemetry CSV helpers.
- `src/moe/checkpoint.rs`
  - GGUF parsing, mmap access, tensor slicing, metadata.
- `src/moe/adapter.rs`
  - Model-family adapter resolution.
- `src/moe/routing.rs`
  - Router math, gate scores, expert selection.
- `src/projector.rs`
  - Spike-to-embedding projection.
- `src/funnel.rs`
  - Telemetry funnel and GIF hidden layer.
- `src/telemetry.rs`
  - `TelemetryEncoder` and `TelemetrySnapshot`.
- `src/latent.rs`
  - SAAQ latent calibration and CSV export.
- `src/gpu/`
  - CUDA/cust wrapper layer and GPU kernels.
- `src/gpu/kernels/`
  - CUDA `.cu` / `.cuh` sources.
- `examples/support/config.rs`
  - Environment/config resolution for examples.
- `examples/support/observability.rs`
  - Shared tracing/Sentry wrapper for example binaries.
- `docs/ARCHITECTURE.md`
  - Architecture, hidden control flow, path behavior.
- `docs/RUN_PROFILES.md`
  - Validated run profiles.
- `docs/PROMOTION_RULES.md`
  - Rules for graduating modules into `rmems-*` crates.
- `docs/MODULE_STATUS.md`
  - Current status of each module.
- `manifests/proven_components.toml`
  - Machine-readable promotion status.
- `manifests/known_good_runs.md`
  - Blessed run ID log.

## Build and validation commands

Prefer the checked-in `justfile` recipes.

Basic commands:

```bash
just setup
cargo check --all-targets --no-default-features
cargo test --no-default-features
```

On CUDA-equipped setups with `nvcc` available, `just check` and `just test`
exercise the default feature set.

## Entry Order

- Start at `src/lib.rs` for the exported crate surface.
- Read `src/model/mod.rs` before descending into `core.rs` and `temporal.rs`.
- Read `src/moe/mod.rs` before descending into `adapter.rs`, `checkpoint.rs`, and `routing.rs`.
- Use `examples/saaq_latent_calibration.rs` as the research loop entrypoint.

## Workflow Policy

- All `corinth-canal` work stays in the repository root.
- Do not create or use additional `corinth-canal` worktrees outside the main repo.
- External edits are limited to approved research tools and dependencies unless explicitly requested.

## Git Workflow

- Prefer `git` commands over MCP tools for branch and PR operations in this environment.
- Keep behavioral changes separate from structural refactors when possible.
- Run `cargo check --no-default-features` and `cargo test --no-default-features` before closing substantial Rust changes.
- On CUDA-equipped setups with `nvcc` available, also run the default-feature path and `cargo build --examples`.

## Repository Context

- **Repo**: `rmems/corinth-canal`
- **Main branch**: `main`
- **Language**: Rust (edition 2024)
- **Key concepts**: SNN-LLM hybrid, GGUF checkpoints, MoE routing, GPU temporal simulation, SAAQ validation
