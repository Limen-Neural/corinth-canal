# Module Status

Live snapshot of every `src/` module's position on the promotion ladder.
Machine-readable mirror: `manifests/proven_components.toml`.

Status legend: `reference` · `stabilizing` · `proven` · `frozen`
(see `docs/PROMOTION_RULES.md`).

| Module | Status | Target `rmems` crate | Notes |
|--------|--------|----------------------|-------|
| `src/model/core.rs` | reference | `rmems-model` | Orchestration layer still couples runtime behavior, artifact wiring, and GGUF-backed routing. Not yet ready to promote as an isolated surface. |
| `src/model/temporal.rs` | stabilizing | `rmems-model` | GPU temporal loop is tight and proven. Legacy fallback to a CWD-relative routing CSV still exists when `ModelConfig::gpu_routing_telemetry_path` is unset, so callers must keep the sink explicit. |
| `src/model/telemetry_io.rs` | stabilizing | `rmems-model` | Pure helper. Public behavior is stable; promotion depends mainly on the surrounding runtime/API cleanup. |
| `src/moe/mod.rs` | stabilizing | `rmems-moe` | Host entry for `OlmoeRouter`; surface is clean. Pending full model-family validation matrix. |
| `src/moe/checkpoint.rs` | reference | `rmems-moe` | GGUF parser/mmap/dequant layer works, but still needs a broader parser and format test battery before promotion. |
| `src/moe/adapter.rs` | stabilizing | `rmems-moe` | Five-family adapter resolution is implemented; needs broader validation coverage. |
| `src/moe/routing.rs` | stabilizing | `rmems-moe` | Stateless routing math. Low-risk promotion candidate. |
| `src/projector.rs` | stabilizing | `rmems-projector` | `ProjectionMode` surface is stable; `SpikingTernary` remains the live research path. |
| `src/funnel.rs` | reference | `rmems-funnel` | CPU GIF hidden layer is still shared with the broader runtime and validation path. |
| `src/telemetry.rs` | stabilizing | `rmems-telemetry` | Telemetry encoding surface is small and stable. `TelemetrySnapshot` now includes heartbeat fields in addition to physical telemetry channels. |
| `src/latent.rs` | stabilizing | `rmems-latent` | Dual-SAAQ emission is in place. Determinism and campaign validation remain the main graduation gate. |
| `src/gpu/*` | reference | `rmems-gpu` | Kernel sources and cust wrappers remain coupled to the reference repo build/runtime assumptions. Promotion is still blocked on portability and validation breadth. |
| `examples/support/config.rs` | reference | n/a | Intentionally stays here — it is the env-truth surface for the reference repo only. |

## Known blockers

- **Machine-local discovery root.** `examples/support/mod.rs` walks
  `$HOME/Downloads/SNN_Quantization`. Fine for the reference repo; must
  not be copied into any `rmems` crate.
- **Legacy routing-CSV fallback.** The runtime now supports
  `ModelConfig::gpu_routing_telemetry_path`, but any caller that leaves it
  unset still falls back to the CWD-relative filename
  `snn_gpu_routing_telemetry.csv`.
- **`build.rs` fatbin compilation.** Assumes nvcc + `sm_120` targets on the
  author's box. `gpu-stub` covers the CI / non-CUDA case.
