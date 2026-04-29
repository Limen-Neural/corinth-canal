set dotenv-load := true
set dotenv-filename := ".env.local"

# Default: list all recipes.
default:
    @just --list

# One-time setup sanity check: verify every doc + manifest exists.
setup:
    @test -f .env.local || echo "warn: .env.local missing (copy from .env.example)"
    @test -d artifacts  || mkdir -p artifacts
    @echo "ok: scaffolding present"

# Fast compile sweep.
check:
    cargo check --all-targets

# Full test suite (CPU-only paths; GPU tests gated on hardware).
test:
    cargo test

# GPU smoke test — 10k direct GPU ticks against a real GGUF checkpoint.
# Requires GGUF_CHECKPOINT_PATH in .env.local.
smoke:
    cargo run --release --example gpu_smoke_test

# CSV replay demo.
#   just replay PATH=/path/to/telemetry.csv
replay PATH:
    cargo run --release --example csv_replay -- {{PATH}}

# Full SAAQ latent calibration sweep using current .env.local values.
saaq:
    cargo run --release --example saaq_latent_calibration

# Phases: synthetic/heartbeat-off, csv/heartbeat-off, csv/heartbeat-on.
# Reads LINEUP_CONFIG and (for phases 2-3) TELEMETRY_CSV_PATH from .env.local.
# Falls back to configs/saaq15_moe_lineup.toml when LINEUP_CONFIG is unset.
# Full SAAQ 1.5 MoE baseline campaign (3 phases x REPEAT_COUNT runs per model).
saaq-campaign:
    @echo ">>> phase 1/3: synthetic, heartbeat off, repeat=2"
    LINEUP_CONFIG="${LINEUP_CONFIG:-configs/saaq15_moe_lineup.toml}" \
        SAAQ_RULE=saaq_v1_5 REPEAT_COUNT=2 TELEMETRY_SOURCE=synthetic \
        HEARTBEAT_MATRIX=off RUN_TAG=campaign_syn_off \
        cargo run --release --example saaq_latent_calibration
    @echo ">>> phase 2/3: csv, heartbeat off, repeat=2"
    LINEUP_CONFIG="${LINEUP_CONFIG:-configs/saaq15_moe_lineup.toml}" \
        SAAQ_RULE=saaq_v1_5 REPEAT_COUNT=2 TELEMETRY_SOURCE=csv \
        HEARTBEAT_MATRIX=off RUN_TAG=campaign_csv_off \
        cargo run --release --example saaq_latent_calibration
    @echo ">>> phase 3/3: csv, heartbeat on, repeat=2"
    LINEUP_CONFIG="${LINEUP_CONFIG:-configs/saaq15_moe_lineup.toml}" \
        SAAQ_RULE=saaq_v1_5 REPEAT_COUNT=2 TELEMETRY_SOURCE=csv \
        HEARTBEAT_MATRIX=on  RUN_TAG=campaign_csv_on \
        cargo run --release --example saaq_latent_calibration
    @echo "ok: campaign finished; see artifacts/index.csv"

# Force CSV-replay mode for the SAAQ sweep. TELEMETRY_CSV_PATH must be set
# in the environment or passed explicitly:
#   just saaq-csv TELEMETRY_CSV_PATH=/path/to/telemetry.csv
saaq-csv:
    TELEMETRY_SOURCE=csv cargo run --release --example saaq_latent_calibration

# Matrix sweep: both heartbeat modes, both SAAQ rules (dual emission).
saaq-sweep:
    HEARTBEAT_MATRIX= cargo run --release --example saaq_latent_calibration

# Telemetry bridge demo (routing_mode switchable via ROUTING_MODE env).
bridge:
    cargo run --release --example telemetry_bridge

# Probe the configured lineup (LINEUP_CONFIG / GGUF_CHECKPOINT_PATH /
# autodiscovery) and print the preferred GPU synapse tensor + ggml_type per
# model. Writes <output_root>/synapse_diagnostic.json. No SAAQ ticks, no
# heartbeat, no campaign side-effects (issue #31).
synapse-diag:
    cargo run --release --example synapse_diagnostic

# Wipe everything under ./artifacts except the .gitkeep anchor.
clean-artifacts:
    find artifacts -mindepth 1 ! -name .gitkeep -exec rm -rf {} +
    @echo "ok: artifacts/ emptied"
