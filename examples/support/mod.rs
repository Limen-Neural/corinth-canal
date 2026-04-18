//! Shared helper functions for the example binaries.
#![allow(dead_code)]

use corinth_canal::{
    model::{Model, ModelConfig},
    moe::RoutingMode,
    projector::ProjectionMode,
    EMBEDDING_DIM,
};
use std::io::Error;

pub const DEFAULT_MATH_PROMPT_TOKEN_IDS: [usize; 9] =
    [402, 11492, 286, 257, 4568, 318, 12056, 4202, 13];

pub fn gguf_checkpoint_path_or_default() -> String {
    std::env::var("GGUF_CHECKPOINT_PATH").unwrap_or_default()
}

pub fn required_gguf_checkpoint_path() -> Result<String, Box<dyn std::error::Error>> {
    let model_path = gguf_checkpoint_path_or_default();
    if model_path.trim().is_empty() {
        return Err(Error::other("GGUF_CHECKPOINT_PATH must point to a GGUF checkpoint").into());
    }
    Ok(model_path)
}

pub fn default_spiking_model_config(gguf_checkpoint_path: String, snn_steps: usize) -> ModelConfig {
    ModelConfig {
        gguf_checkpoint_path,
        gpu_synapse_tensor_name: "blk.0.attn_q.weight".into(),
        num_experts: 8,
        top_k_experts: 1,
        routing_mode: RoutingMode::SpikingSim,
        snn_steps,
        projection_mode: ProjectionMode::SpikingTernary,
    }
}

pub fn mean_pool_prompt_embeddings(
    model: &mut Model,
    token_ids: &[usize],
) -> corinth_canal::Result<Vec<f32>> {
    let mut pooled = vec![0.0f32; EMBEDDING_DIM];

    for &token in token_ids {
        let emb = model.extract_token_embedding(token)?;
        for (dst, src) in pooled.iter_mut().zip(emb.iter()) {
            *dst += *src;
        }
    }

    for value in &mut pooled {
        *value /= token_ids.len().max(1) as f32;
    }

    let l2_norm = pooled.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if l2_norm > 1e-8 {
        for value in &mut pooled {
            *value /= l2_norm;
        }
    }

    Ok(pooled)
}
