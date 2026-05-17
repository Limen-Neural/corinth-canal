//! # corinth-canal
//!
//! `corinth-canal` is the single-crate reference implementation for the `rmems`
//! SNN-logic quantization line.
//!
//! The crate keeps the full telemetry -> spiking -> projector -> router path in
//! one repository so the research loop can be exercised end to end before proven
//! components are promoted into separate `rmems-*` crates.
//!
//! ## Runtime pipeline
//!
//! ```text
//! TelemetrySnapshot
//!        │
//!        ▼  TelemetryEncoder (CPU) / project_snapshot_current (GPU)
//! ternary telemetry events (+1 / 0 / -1 per channel)
//!        │
//!        ▼  SignedSplitBankBridge (CPU) / GPU input_spikes buffer
//! input spike train
//!        │
//!        ▼  SparseGifHiddenLayer (CPU) / gif_step_weighted_tick (GPU)
//! hidden spike train + membrane potentials
//!        │
//!        ▼  Projector
//! dense embedding [EMBEDDING_DIM = 2048]
//!        │
//!        ▼  OlmoeRouter
//! expert_weights + selected_experts + routed hidden state
//!        │
//!        ▼  SAAQ latent calibration / telemetry export
//! ```
//!
//! ## Model loading
//!
//! The routing bridge is GGUF-backed and custom to this repository.
//! `OlmoeRouter` parses GGUF checkpoints in-repo, resolves a supported model
//! family, extracts routing tensors and token embeddings, and selects a GPU
//! synapse source from one of:
//!
//! - real `F16`
//! - dequantized `Q8_0`
//! - dequantized `Q5_K`
//! - synthetic fallback
//!
//! Supported families in code today are `Olmoe`, `Qwen3Moe`, `Gemma4`,
//! `DeepSeek2`, and `LlamaMoe`.
//!
//! ## Quick start
//!
//! The `model` module is available when the `cuda` feature is enabled (the
//! default feature set).
//!
//! ```no_run
//! # #[cfg(feature = "cuda")] {
//! use corinth_canal::model::{Model, ModelConfig};
//! use corinth_canal::TelemetrySnapshot;
//!
//! let mut model = Model::new(ModelConfig::default()).unwrap();
//! let output = model.forward(&TelemetrySnapshot::default()).unwrap();
//!
//! println!("Selected experts: {:?}", output.selected_experts);
//! # }
//! ```

pub mod error;
pub mod funnel;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod heartbeat;
pub mod latent;
#[cfg(feature = "cuda")]
pub mod model;
pub mod moe;
pub mod projector;
pub mod telemetry;
pub mod types;

pub use error::{HybridError, Result};
pub use funnel::{
    FUNNEL_HIDDEN_NEURONS, FUNNEL_INPUT_NEURONS, FunnelActivity, SignedSplitBankBridge,
    SparseGifHiddenLayer, TelemetryFunnel,
};
pub use heartbeat::HeartbeatInjector;
pub use latent::{
    SaaqUpdateRule, SnnDualLatentCalibrator, SnnLatentCalibrator, SnnLatentCsvExporter,
    SnnLatentSnapshot,
};
pub use telemetry::TelemetryEncoder;
pub use types::{EMBEDDING_DIM, HeartbeatConfig, ModelFamily, TelemetrySnapshot};

pub mod tensor;

// New folder name metric, came out of triple code duplication
pub(crate) mod metric;
