# Architecture

`corinth-canal` is the single-crate reference implementation of the `rmems`
SNN-logic quantization bridge. It intentionally keeps the telemetry encoder,
spiking hidden layer, projector, GGUF-backed routing bridge, and validation
artifacts in one repository so the full research loop can be exercised before
proven components are promoted into separate `rmems-*` crates.

## Block diagram

```text
TelemetrySnapshot
       │
       ▼  TelemetryEncoder (CPU) / project_snapshot_current (GPU)
[i8; 4] ternary telemetry events (+1 / 0 / -1)
       │
       ▼  SignedSplitBankBridge (CPU) / GPU input_spikes buffer
input spike train
       │
       ▼  SparseGifHiddenLayer (CPU) / gif_step_weighted_tick (GPU)
hidden spike train + membrane/adaptation state
       │
       ▼  Projector
embedding [EMBEDDING_DIM = 2048]
       │
       ▼  OlmoeRouter (stub | dense | spiking sim)
expert_weights + selected_experts + routed hidden state
       │
       ▼  SAAQ latent calibration / telemetry export
```

## CPU and GPU paths

### CPU path

The CPU path is assembled from pure Rust components:

- `TelemetryEncoder` converts a `TelemetrySnapshot` into `[i8; 4]` ternary
  events using per-channel delta thresholds.
- `SignedSplitBankBridge` expands those ternary events into an input spike
  train.
- `SparseGifHiddenLayer` runs a 2048-neuron GIF hidden layer with adaptive
  thresholds.
- `Projector` converts hidden activity into a 2048-dimensional embedding.
- `OlmoeRouter` consumes the embedding and produces expert weights / selected
  experts.

### GPU path

The GPU temporal path is orchestrated by `Model::prepare_gpu_temporal`,
`Model::tick_gpu_temporal`, and `Model::forward_gpu_temporal`.

Key pieces:

- `project_snapshot_current` projects 4-channel telemetry into the GPU temporal
  input buffer.
- `gif_step_weighted_tick` advances the resident GIF temporal state.
- GPU synapse weights are loaded from a GGUF-backed source when available.
- The GPU path reuses resident synapse weights across calls using a source
  signature cached in `GpuAccelerator`.
- `forward_gpu_temporal` writes routing telemetry through the shared CSV helper.
- `tick_gpu_temporal` advances GPU state but does not itself append routing CSV
  rows.

## Module map

| Module | Role |
|--------|------|
| `src/model/core.rs` | Runtime orchestration, config validation, forward paths |
| `src/model/temporal.rs` | GPU temporal loop (`prepare_gpu_temporal`, `tick_gpu_temporal`, `forward_gpu_temporal`) |
| `src/model/telemetry_io.rs` | Shared CSV writer for GPU routing telemetry |
| `src/moe/mod.rs` | `OlmoeRouter` host with routing-mode dispatch |
| `src/moe/checkpoint.rs` | GGUF parse, mmap, tensor slicing, dequantization helpers |
| `src/moe/adapter.rs` | Model-family adapter resolution and tensor selection |
| `src/moe/routing.rs` | Router math (gate scores, resampling, normalization, top-k) |
| `src/projector.rs` | `ProjectionMode` spike-to-embedding projection |
| `src/funnel.rs` | Telemetry funnel, signed split banks, GIF hidden layer |
| `src/telemetry.rs` | `TelemetryEncoder` and `TelemetrySnapshot` bridge |
| `src/latent.rs` | SAAQ 1.0 / 1.5 calibration and CSV export |
| `src/gpu/` | CUDA wrappers, buffers, kernel launchers |
| `examples/support/config.rs` | Example-only environment/config resolution |

## Model loading and routing bridge

The model-loading interface is custom and GGUF-backed.

`OlmoeRouter`:

- memory-maps GGUF checkpoints in-repo
- resolves a supported model family from GGUF metadata
- locates the routing tensor and token embedding tensor
- exposes token embedding extraction for validation workflows
- selects a GPU synapse source from one of:
  - real `F16`
  - dequantized `Q8_0`
  - dequantized `Q5_K`
  - synthetic fallback

Supported families in code today:

- `Olmoe`
- `Qwen3Moe`
- `Gemma4`
- `DeepSeek2`
- `LlamaMoe`

## Routing / projection modes

### Projection modes

- `RateSum`
- `TemporalHistogram`
- `MembraneSnapshot`
- `SpikingTernary`

### Routing modes

- `StubUniform`
- `DenseSim`
- `SpikingSim`

## Validation entrypoint and artifacts

The primary research/validation loop is `examples/saaq_latent_calibration.rs`.
For each run it writes artifacts including:

- `tick_telemetry.txt`
- `latent_telemetry.csv`
- `run_manifest.json`
- `summary.json`
- `snn_gpu_routing_telemetry.csv`

The GPU routing telemetry CSV schema is:

```text
token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction
```

The latent telemetry CSV includes both SAAQ trajectories via
`SnnDualLatentCalibrator`; one rule is selected as the primary/legacy
compatibility projection while the legacy and v1.5 columns are both emitted.

## Hidden control flow

A few control paths are easy to miss from a top-level read.

### Routing telemetry CSV path behavior

`Model::forward_gpu_temporal` and `Model::compute_routing_telemetry` resolve the
routing telemetry sink through `ModelConfig::gpu_routing_telemetry_path`.
When this field is `None`, the runtime falls back to the legacy
CWD-relative filename `snn_gpu_routing_telemetry.csv`.

Implications:

- `examples/saaq_latent_calibration.rs` sets the path explicitly into the run
  directory, so its routing telemetry stays per-run.
- Other call sites may still rely on the legacy fallback when they do not set
  the path explicitly.

### Env-resolved paths

Machine-specific path discovery belongs in `examples/support/config.rs`.
The library code under `src/` does not perform environment-variable path
resolution for checkpoints, telemetry CSVs, or artifact roots.

### Telemetry source stamping

The validation runner stamps a source label into `run_manifest.json` using one
of:

- `synthetic`
- `synthetic_fallback`
- `csv_<stem>`

This makes fallback behavior explicit in artifacts instead of hiding it behind a
successful run.

## Observability

The example binaries share observability helpers under
`examples/support/observability.rs`.

- `command_start` and `command_finish` tracing events are emitted for every
  example command.
- Sentry is opt-in only. If `SENTRY_DSN` is unset or blank, the examples remain
  local/offline.
- The wrappers attach only safe diagnostic fields such as `repo`, `command`,
  `git_sha`, `model_slug`, `telemetry_source`, `heartbeat_enabled`,
  `validation_status`, and `error_category`.
- Absolute checkpoint paths and artifact paths are not attached as Sentry tags
  by the wrappers.
