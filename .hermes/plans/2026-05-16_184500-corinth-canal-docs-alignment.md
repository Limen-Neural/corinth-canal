# Corinth-canal docs alignment plan

## Goal

Prepare a docs-only follow-up that aligns repository documentation with the current corinth-canal implementation, especially the architecture/runtime/model-loading/output details surfaced during the code review and reflected in GitHub issues #49 and #50.

## Current context / assumptions

- This is a single-crate Rust repository and should remain documented as such.
- The request is planning only. No code changes, no issue edits, no PR creation in this plan.
- Source-level review already identified likely documentation drift in:
  - `src/lib.rs`
  - `docs/ARCHITECTURE.md`
  - `docs/MODULE_STATUS.md`
  - `docs/RUN_PROFILES.md`
- Reviewed implementation indicates the docs should reflect:
  - custom GGUF-backed model loading
  - CPU and GPU execution paths
  - SAAQ dual emission
  - heartbeat/telemetry semantics
  - routing telemetry outputs and run artifacts
- Any claims not directly grounded in source should be marked `needs verification`.

## Proposed approach

Do a docs-only pass in two layers:

1. Reconcile existing core docs with the actual implementation.
2. Add narrowly scoped supporting docs only if the existing files become overloaded.

The first pass should favor updating existing authoritative docs before adding new files.

## Implementation-grounded facts the docs should reflect

### Runtime pipeline

The current top-level flow to document is:

`TelemetrySnapshot`
`-> TelemetryEncoder / GPU snapshot projection`
`-> ternary spikes`
`-> SignedSplitBankBridge or GPU input_spikes`
`-> SparseGifHiddenLayer or GPU GIF temporal loop`
`-> Projector`
`-> OlmoeRouter`
`-> SAAQ latent calibration / telemetry outputs`

### Model loading / routing

- Model loading is custom and GGUF-backed.
- Supported families in code:
  - `Olmoe`
  - `Qwen3Moe`
  - `Gemma4`
  - `DeepSeek2`
  - `LlamaMoe`
- Synapse source cascade:
  - real `F16`
  - dequantized `Q8_0`
  - dequantized `Q5_K`
  - synthetic fallback
- Routing modes:
  - `StubUniform`
  - `DenseSim`
  - `SpikingSim`

### Projection / telemetry / calibration

- Projection modes:
  - `RateSum`
  - `TemporalHistogram`
  - `MembraneSnapshot`
  - `SpikingTernary`
- `TelemetrySnapshot` includes:
  - `gpu_temp_c`
  - `gpu_power_w`
  - `cpu_tctl_c`
  - `cpu_package_power_w`
  - `heartbeat_signal`
  - `heartbeat_enabled`
  - `timestamp_ms`
- Dual SAAQ emission exists in latent CSV output.
- GPU routing telemetry CSV schema is:
  - `token_idx,best_score,best_walker,spike_count,mean_adaptation,active_fraction`

## Step-by-step plan

### Phase 1: Reconfirm source of truth before editing

1. Re-read the exact implementation surfaces that back the docs changes:
   - `src/lib.rs`
   - `src/types.rs`
   - `src/telemetry.rs`
   - `src/funnel.rs`
   - `src/projector.rs`
   - `src/latent.rs`
   - `src/model/core.rs`
   - `src/model/temporal.rs`
   - `src/model/telemetry_io.rs`
   - `src/moe/mod.rs`
   - `src/moe/checkpoint.rs`
   - `src/moe/adapter.rs`
   - `src/moe/routing.rs`
   - `examples/saaq_latent_calibration.rs`
2. Re-read the target docs to identify exact stale wording:
   - `docs/ARCHITECTURE.md`
   - `docs/MODULE_STATUS.md`
   - `docs/RUN_PROFILES.md`
3. Flag every statement that is not directly supported by source as `needs verification` before editing.

### Phase 2: Update existing docs first

4. Update `src/lib.rs` crate-level docs:
   - replace the simplified architecture summary with the actual current pipeline
   - keep the wording concise and implementation-grounded
   - avoid overselling experimental claims

5. Update `docs/ARCHITECTURE.md`:
   - align the block diagram with the actual CPU/GPU path split
   - ensure module roles match current files
   - keep the hidden-control-flow and observability sections only if still source-supported
   - explicitly mention the research entrypoint `examples/saaq_latent_calibration.rs`

6. Update `docs/MODULE_STATUS.md`:
   - reconcile stale notes about GPU routing telemetry path handling
   - preserve caveats that still appear true
   - avoid claiming promotion-readiness beyond what current status files support

7. Update `docs/RUN_PROFILES.md`:
   - retain command-oriented usage
   - add the actual output artifacts each run produces when supported by `examples/saaq_latent_calibration.rs`
   - document the role of `latent_telemetry.csv`, `tick_telemetry.txt`, `run_manifest.json`, `summary.json`, and `snn_gpu_routing_telemetry.csv`

### Phase 3: Add support docs only if still needed

8. Decide whether to add `docs/MODEL_LOADING.md`.
   Add only if the model-loading explanation would make `docs/ARCHITECTURE.md` too dense.
   If added, document:
   - GGUF parsing/mmap flow
   - family inference
   - routing tensor resolution
   - token embedding extraction
   - synapse source cascade

9. Decide whether to add `docs/ARTIFACTS.md`.
   Add only if `docs/RUN_PROFILES.md` becomes too operationally heavy.
   If added, document:
   - artifact names
   - where they are produced
   - what each file contains
   - exact CSV header/schema where known from source

10. Decide whether to add `docs/TERMINOLOGY.md`.
    Add only if terminology remains scattered after the core doc updates.
    Candidate terms:
    - SAAQ
    - heartbeat
    - telemetry source labels
    - routing tensor
    - synapse source
    - bank mapping
    - ternary events
    - best walker

### Phase 4: Final consistency pass

11. Do a consistency pass across the updated docs to ensure:
   - the same pipeline wording is used everywhere
   - routing and projection mode names match source exactly
   - family names match source exactly
   - CSV schema strings match code exactly
   - any uncertainty is clearly labeled `needs verification`

12. Prepare a concise PR summary listing:
   - files changed
   - source-backed facts clarified
   - remaining open questions / `needs verification` items

## Files likely to change

Highest priority:
- `src/lib.rs`
- `docs/ARCHITECTURE.md`
- `docs/MODULE_STATUS.md`
- `docs/RUN_PROFILES.md`

Optional new docs if justified:
- `docs/MODEL_LOADING.md`
- `docs/ARTIFACTS.md`
- `docs/TERMINOLOGY.md`

## Tests / validation

Because this is docs-only work, validation should focus on accuracy and consistency:

1. Re-check every technical claim against source before finalizing.
2. Verify exact identifiers and schema strings against code:
   - projection mode names
   - routing mode names
   - model family names
   - CSV headers
3. Ensure the crate-level doc comment in `src/lib.rs` is syntactically valid Rust doc text.
4. Optional sanity check after docs edit:
   - `cargo check`
   - only to ensure doc comment edits did not accidentally break syntax
   - skip if plan owner wants strictly docs editing without build validation

## Risks, tradeoffs, and open questions

### Risks

- Accidentally carrying over narrative/research claims that are not directly supported by current source.
- Over-documenting experimental internals and making the docs harder to maintain.
- Mixing “what the code does now” with “what the research intends long-term.”

### Tradeoffs

- Updating existing docs keeps the documentation surface smaller and more authoritative.
- Adding new docs improves discoverability for model loading and artifacts, but increases maintenance cost.

### Open questions

- Should `README.md` be updated in the same docs pass? `needs verification`
- Is “neighborhood mapping” an intended project term, or should docs standardize on “bank mapping” / “sparse fan-in”? `needs verification`
- Should experiment narrative docs be revised now, or only the implementation-facing docs? `needs verification`
- Should a docs-only PR include generated examples/snippets, or stay purely descriptive? `needs verification`

## Suggested deliverable shape

A clean docs-only PR with:
- updated core docs
- optional one or two new support docs only if clearly justified
- a PR body that lists exact implementation facts now reflected in docs
- a short `needs verification` section for anything not proven directly from source
