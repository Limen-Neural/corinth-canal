# Run Profiles

Every validated tuple of `(example, telemetry source, heartbeat, SAAQ rule)`
is catalogued here with the exact command. Anything not in this table has
not been blessed and should be considered experimental.

## Prerequisites

- Fill in `.env.local` from `.env.example`.
- `just setup` — sanity check that the scaffolding is in place.

## SAAQ latent calibration (primary research loop)

| Profile | Telemetry | Heartbeat | SAAQ rule | Command |
|---------|-----------|-----------|-----------|---------|
| Smoke, synthetic | synthetic | both | 1.5 | `just saaq` |
| SR.jl corpus, full loop | csv (RE4) | both | 1.5 + 1.0 dual | `TICKS=0 just saaq-csv` |
| Wraparound sanity | csv (RE4) | both | 1.5 | `TICKS=2000 just saaq-csv` |
| Heartbeat OFF only | synthetic | off | 1.5 | `HEARTBEAT_MATRIX=off just saaq` |
| Heartbeat ON only | synthetic | on | 1.5 | `HEARTBEAT_MATRIX=on just saaq` |
| Legacy rule parity | synthetic | both | 1.0 | `SAAQ_RULE=legacy just saaq` |

Dual-SAAQ columns are emitted regardless of `SAAQ_RULE`; the env var only
selects which rule fills the legacy compatibility columns.

### Validation outputs

`examples/saaq_latent_calibration.rs` writes a per-run directory under the
configured output root. Each run directory contains:

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
`SnnDualLatentCalibrator`:

- legacy / primary compatibility columns
- explicit legacy trajectory columns
- explicit v1.5 trajectory columns

### Operational notes

- `TICKS=0` with `TELEMETRY_SOURCE=csv` uses the full CSV row count so corpus
  runs cover exactly one telemetry loop.
- When `TICKS > csv_row_count`, the runner warns about wraparound contamination.
- `STRICT_REPEAT_CHECK=true` enables repeat-to-repeat comparison in the
  validation workflow.
- `run_manifest.json` stamps the actual telemetry source label, including
  `synthetic_fallback` when CSV replay degrades.

## GPU smoke test

| Profile | Command |
|---------|---------|
| 10k GPU ticks, real checkpoint | `GGUF_CHECKPOINT_PATH=... just smoke` |

This example is the direct GPU-temporal smoke path. It is the correct entrypoint
for validating resident synapse upload, GIF weighted temporal stepping, and the
on-device best-walker reduction.

## CSV replay

| Profile | Command |
|---------|---------|
| Ingest canonical telemetry CSV | `just replay /path/to/telemetry.csv` |

Canonical CSV schema consumed by replay and validation:

```text
timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w
```

## Telemetry bridge demo

| Profile | Command |
|---------|---------|
| Spiking routing | `just bridge` |
| Dense routing | `ROUTING_MODE=dense just bridge` |
| Stub routing | `ROUTING_MODE=stub just bridge` |

## Model discovery

If `GGUF_CHECKPOINT_PATH` is unset, `saaq_latent_calibration` auto-discovers
up to five MoE families under `$HOME/Downloads/SNN_Quantization/`:

- `olmoe-0125-gguf/OLMoE-1B-7B-0125-Instruct-F16.gguf`
- `models/qwen3-moe-i1-GGUF/qwen3-moe.i1-IQ3_M.gguf`
- `models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf`
- `models/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q6_K_L.gguf`
- `models/Llama-3.2-8X3B-MOE-Dark-Champion-GGUF/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-q5_k_m.gguf`

This discovery root is a machine-local convention on the author's Fedora
box. CI and contributor machines should set `GGUF_CHECKPOINT_PATH`
explicitly.

## Supported routing/model surfaces reflected in code

Routing modes:

- `StubUniform`
- `DenseSim`
- `SpikingSim`

Model families supported by the GGUF adapter layer:

- `Olmoe`
- `Qwen3Moe`
- `Gemma4`
- `DeepSeek2`
- `LlamaMoe`
