# corinth-canal

Single-crate reference implementation of the `rmems` SNN-logic quantization
bridge.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)

## Overview

`corinth-canal` keeps the telemetry encoder, spiking hidden layer, projector,
GGUF-backed routing bridge, and SAAQ validation loop in one repository so the
full research path can be exercised end to end.

The current runtime flow is:

```text
TelemetrySnapshot
       |
       v  TelemetryEncoder (CPU) / project_snapshot_current (GPU)
ternary telemetry events (+1 / 0 / -1)
       |
       v  SignedSplitBankBridge (CPU) / GPU input_spikes
input spike train
       |
       v  SparseGifHiddenLayer (CPU) / gif_step_weighted_tick (GPU)
hidden spike train + membrane state
       |
       v  Projector
embedding [2048]
       |
       v  OlmoeRouter
expert_weights + selected_experts + routed hidden state
       |
       v  SAAQ latent calibration / telemetry export
```

## Scope

This repository is intentionally kept as a single crate. Proven components are
expected to graduate later into `rmems-*` crates according to
`docs/PROMOTION_RULES.md`, but `corinth-canal` itself remains the reference repo
for the full end-to-end loop.

## Implementation highlights

### Custom GGUF-backed routing bridge

The model-loading interface is custom to this repository.

`OlmoeRouter`:

- parses GGUF checkpoints in-repo
- resolves a supported model family from checkpoint metadata
- extracts routing tensors and token embeddings
- selects a GPU synapse source from one of:
  - real `F16`
  - dequantized `Q8_0`
  - dequantized `Q5_K`
  - synthetic fallback

Supported families in code:

- `Olmoe`
- `Qwen3Moe`
- `Gemma4`
- `DeepSeek2`
- `LlamaMoe`

### Projection modes

- `RateSum`
- `TemporalHistogram`
- `MembraneSnapshot`
- `SpikingTernary`

### Routing modes

- `StubUniform`
- `DenseSim`
- `SpikingSim`

### Telemetry / heartbeat

`TelemetrySnapshot` carries:

- `gpu_temp_c`
- `gpu_power_w`
- `cpu_tctl_c`
- `cpu_package_power_w`
- `heartbeat_signal`
- `heartbeat_enabled`
- `timestamp_ms`

### SAAQ calibration

The validation path uses `SnnDualLatentCalibrator`, which emits both the legacy
and v1.5 SAAQ trajectories in latent telemetry output while preserving legacy
compatibility columns.

## Primary docs

- `docs/ARCHITECTURE.md` — runtime architecture, module map, hidden control flow
- `docs/RUN_PROFILES.md` — validated commands and per-run outputs
- `docs/PROMOTION_RULES.md` — module promotion rules
- `docs/MODULE_STATUS.md` — current module stability/promotion status
- `manifests/proven_components.toml` — machine-readable module status mirror

## Main entrypoints

- `examples/saaq_latent_calibration.rs` — primary research / validation loop
- `examples/gpu_smoke_test.rs` — GPU temporal smoke validation
- `examples/csv_replay.rs` — canonical telemetry CSV replay
- `examples/telemetry_bridge.rs` — routing demonstration

## Validation outputs

The validation runner writes per-run artifacts including:

- `tick_telemetry.txt`
- `latent_telemetry.csv`
- `run_manifest.json`
- `summary.json`
- `snn_gpu_routing_telemetry.csv`

GPU routing telemetry CSV schema:

```text
token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction
```

## Quick start

### Check the repo

```bash
just setup
just check
just test
```

### Run the primary validation loop

```bash
just saaq
```

### Run the GPU smoke path with a real checkpoint

```bash
GGUF_CHECKPOINT_PATH=/path/to/model.gguf just smoke
```

## Notes

- CPU-only buildability is preserved.
- CUDA/GPU behavior is preserved behind the `cuda` feature.
- Machine-local checkpoint discovery under `$HOME/Downloads/SNN_Quantization`
  is a reference-repo convention, not a portable interface.
