# GIF Hidden Layer Zero-Firing Diagnosis

## Root Cause

The `avg_pop_firing_rate_hz` is **0.0** across all 18 campaign runs because the GIF neurons never reach their firing threshold. This is a **signal magnitude mismatch**, not a logic bug.

### Signal Path Analysis

```
prompt_text → synthetic_text_embedding() → L2 normalize → per-element ≈ 0.022
                                                              ↓
                                                     × heartbeat_gain (0.15–1.24)
                                                              ↓
                                                     input_spikes ≈ [0.003 … 0.027]
                                                              ↓
                                             GPU: drive = dot(weight_row[2048], input_spikes[2048])
                                                              ↓
                                             drive ≈ 0.0 (mixed-sign dot product cancels)
                                                              ↓
                                             membrane = membrane * 0.92 + drive * 0.75
                                                              ↓
                                             threshold = 0.65 + adaptation * 0.22
                                                              ↓
                                             NEVER FIRES ❌
```

### Why the Drive is Too Small

| Parameter | Value | Impact |
|-----------|-------|--------|
| `synthetic_text_embedding()` output | L2-normalized, dim=2048 | Per-element ≈ 1/√2048 ≈ **0.022** |
| `heartbeat_gain()` | 0.15 – 1.24 | Scales to max **0.027** |
| Dot product (2048 terms) | Mixed-sign GGUF weights × tiny inputs | Tends toward **≈ 0** by CLT |
| `GIF_DRIVE_SCALE` | 0.75 | Further reduces already-small drive |
| Steady-state amplification | drive / (1 - GIF_LEAK) = drive × 9.375 | Even 9.375× of ≈0 = ≈0 |
| `GIF_THRESHOLD_BASE` | 0.65 | Unreachable with ≈0 drive |

### The Membrane Data Confirms This

```
tick=2  membrane_dv_dt=0.930  (initial transient from project_snapshot_current)
tick=12 membrane_dv_dt=0.387  (decaying toward steady state)
tick=500 membrane_dv_dt=-0.065 (settled well below threshold, slightly decaying)
```

The membrane rises from `project_snapshot_current` (which injects ~0.9–1.35 per neuron from telemetry), but **the GIF kernel's weighted input from `input_spikes` is negligible** compared to the leak decay. The `project_snapshot_current` kernel runs only in `forward_gpu_temporal()`, NOT in `tick_gpu_temporal()` which is what the SAAQ example uses.

> [!IMPORTANT]
> In `tick_gpu_temporal()`, the ONLY input is `input_spikes` — which are the L2-normalized prompt embedding × heartbeat gain. There is **no snapshot projection** adding external current. The neurons receive only the tiny dot-product drive.

## Fix: Add Input Drive Gain

Add an environment-configurable `INPUT_DRIVE_GAIN` multiplier that scales the prompt embedding **before** it enters the GPU temporal loop:

```rust
// examples/support/mod.rs
pub fn input_drive_gain_from_env() -> f32 {
    env_f32("INPUT_DRIVE_GAIN", 32.0)
}
```

```rust
// saaq_latent_calibration.rs — tick loop
let drive_gain = input_drive_gain_from_env();
let input_spikes: Vec<f32> = prompt_embedding.iter().map(|v| v * gain * drive_gain).collect();
```

### Why 32.0?

- L2-normalized 2048-dim: per-element ≈ 0.022
- After 32× gain: per-element ≈ 0.7
- Dot product of 2048 terms at ±0.7 × weight ≈ ±0.03 each → drive std ≈ 1.4
- With `GIF_DRIVE_SCALE=0.75`: effective drive ≈ 1.05
- Steady-state membrane ≈ 1.05 / 0.08 ≈ 13.1 → **well above threshold 0.65** ✅
- Adaptation feedback will self-regulate firing rate to a healthy ~5-15%

### What This Does NOT Change

- ❌ SAAQ math (latent.rs) — untouched
- ❌ Router math (routing.rs) — untouched  
- ❌ CUDA kernels (spiking_network.cu) — untouched
- ❌ GIF parameters — untouched
- ❌ Heartbeat behavior — untouched
- ❌ CSV schemas — untouched

This is purely **input signal conditioning** — the SNN equivalent of adjusting microphone gain before feeding an audio signal into a neural network.
