//! Routing math and embedding resampling for the GGUF router bridge.

use super::checkpoint::{GgufTensorInfo, MappedGgufCheckpoint};
use crate::error::{HybridError, Result};
use crate::types::EMBEDDING_DIM;
use std::cmp::Ordering;

pub(super) fn checkpoint_gate_scores(
    checkpoint: &MappedGgufCheckpoint,
    model_path: &str,
    routing_tensor_name: &str,
    num_experts: usize,
    embedding: &[f32],
) -> Result<Vec<f32>> {
    let info = checkpoint.tensor_info(routing_tensor_name, model_path)?;
    let weights = checkpoint.f32_tensor(routing_tensor_name, model_path)?;
    let mut gate_scores = Vec::with_capacity(num_experts);
    for expert_id in 0..num_experts {
        let mut score = 0.0f32;
        for (dim, &value) in embedding.iter().enumerate() {
            let index = routing_weight_index(info, expert_id, dim, num_experts, embedding.len())?;
            score += weights[index] * value;
        }
        gate_scores.push(score);
    }
    Ok(gate_scores)
}

pub(super) fn synthetic_gate_scores(num_experts: usize, embedding: &[f32]) -> Vec<f32> {
    let width = embedding.len().max(1);
    let chunk = (width / num_experts.max(1)).max(1);
    let mut gate_scores = Vec::with_capacity(num_experts);
    for expert_id in 0..num_experts {
        let start = (expert_id * chunk) % width;
        let end = (start + chunk).min(width);
        gate_scores.push(embedding[start..end].iter().sum());
    }
    gate_scores
}

pub(super) fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_scores: Vec<f32> = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    let sum_exp: f32 = exp_scores.iter().sum();
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }
    exp_scores
        .into_iter()
        .map(|value| value / sum_exp)
        .collect()
}

pub(super) fn top_k_indices(weights: &[f32], top_k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed
        .into_iter()
        .take(top_k)
        .map(|(idx, _)| idx)
        .collect()
}

pub(super) fn resample_embedding(input: &[f32], target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if input.len() == target_len {
        return input.to_vec();
    }
    if input.is_empty() {
        return vec![0.0; target_len];
    }
    if target_len == 1 {
        return vec![input.iter().sum::<f32>() / input.len() as f32];
    }

    let scale = (input.len() - 1) as f32 / (target_len - 1) as f32;
    let mut out = Vec::with_capacity(target_len);
    for idx in 0..target_len {
        let source = idx as f32 * scale;
        let lo = source.floor() as usize;
        let hi = source.ceil().min((input.len() - 1) as f32) as usize;
        if lo == hi {
            out.push(input[lo]);
            continue;
        }
        let t = source - lo as f32;
        out.push(input[lo] * (1.0 - t) + input[hi] * t);
    }
    out
}

pub(super) fn normalize_l2(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for value in values {
            *value /= norm;
        }
    }
}

pub(super) fn normalize_to_internal_embedding_dim(input: &[f32]) -> Vec<f32> {
    let mut out = resample_embedding(input, EMBEDDING_DIM);
    normalize_l2(&mut out);
    out
}

fn routing_weight_index(
    tensor: &GgufTensorInfo,
    expert_id: usize,
    dim: usize,
    num_experts: usize,
    hidden_size: usize,
) -> Result<usize> {
    let d0 = tensor.dims[0];
    let d1 = tensor.dims[1];

    if d0 == hidden_size && d1 >= num_experts {
        return Ok(dim * d1 + expert_id);
    }
    if d0 >= num_experts && d1 == hidden_size {
        return Ok(expert_id * d1 + dim);
    }

    Err(HybridError::UnsupportedFormat(format!(
        "unsupported routing tensor orientation {:?} for hidden_size={hidden_size}",
        tensor.dims
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_embedding_preserves_target_length() {
        let input = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample_embedding(&input, 8);
        assert_eq!(out.len(), 8);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[7] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_to_internal_dim_returns_2048() {
        let input = vec![0.5; 3072];
        let out = normalize_to_internal_embedding_dim(&input);
        assert_eq!(out.len(), EMBEDDING_DIM);
    }
}
