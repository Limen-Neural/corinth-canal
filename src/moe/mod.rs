//! Public MoE router API backed by a first-block GGUF bridge.
//!
//! This file keeps the public router API small. Private helpers live in
//! `moe/checkpoint.rs` for GGUF parsing + mapped tensor access and
//! `moe/routing.rs` for gate scoring and spiking-state math.

mod checkpoint;
mod routing;

use self::checkpoint::{
    MappedOlmoeCheckpoint, extract_token_embedding_from_checkpoint, probe_and_map_checkpoint,
};
use self::routing::{checkpoint_gate_scores, softmax, synthetic_gate_scores, top_k_indices};
use crate::error::{HybridError, Result};
use crate::types::EMBEDDING_DIM;
pub use crate::types::RoutingMode;

pub(super) const OLMOE_HIDDEN: usize = 2048;
pub(super) const OLMOE_NUM_EXPERTS: usize = 64;
pub(super) const OLMOE_NUM_LAYERS: usize = 16;
pub(super) const ROUTING_TENSOR_NAME: &str = "blk.0.ffn_gate_inp.weight";
pub(super) const DEFAULT_GPU_SYNAPSE_TENSOR_NAME: &str = "blk.0.attn_q.weight";
pub(super) const GGUF_MAGIC: [u8; 4] = [b'G', b'G', b'U', b'F'];
pub(super) const GGUF_VERSION: u32 = 3;
pub(super) const GGML_TYPE_F32: u32 = 0;
pub(super) const GGML_TYPE_F16: u32 = 1;
pub(super) const GGUF_VALUE_TYPE_UINT8: u32 = 0;
pub(super) const GGUF_VALUE_TYPE_INT8: u32 = 1;
pub(super) const GGUF_VALUE_TYPE_UINT16: u32 = 2;
pub(super) const GGUF_VALUE_TYPE_INT16: u32 = 3;
pub(super) const GGUF_VALUE_TYPE_UINT32: u32 = 4;
pub(super) const GGUF_VALUE_TYPE_INT32: u32 = 5;
pub(super) const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
pub(super) const GGUF_VALUE_TYPE_BOOL: u32 = 7;
pub(super) const GGUF_VALUE_TYPE_STRING: u32 = 8;
pub(super) const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
pub(super) const GGUF_VALUE_TYPE_UINT64: u32 = 10;
pub(super) const GGUF_VALUE_TYPE_INT64: u32 = 11;
pub(super) const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

pub struct OlmoeRouter {
    model_path: String,
    num_experts: usize,
    top_k: usize,
    loaded: bool,
    metadata: OlmoeMetadata,
    routing_mode: RoutingMode,
    expert_membranes: Vec<f32>,
    hidden_membranes: Vec<f32>,
    threshold: f32,
    decay: f32,
    checkpoint: Option<MappedOlmoeCheckpoint>,
}

#[derive(Debug, Clone, Default)]
pub(super) struct OlmoeMetadata {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub quantization: String,
}

#[derive(Debug, Clone)]
pub struct OlmoeOutput {
    pub expert_weights: Vec<f32>,
    pub selected_experts: Vec<usize>,
    pub hidden: Vec<f32>,
}

impl OlmoeRouter {
    pub fn load(model_path: &str, num_experts: usize, top_k: usize) -> Result<Self> {
        Self::load_with_mode(
            model_path,
            num_experts,
            top_k,
            RoutingMode::StubUniform,
        )
    }

    pub fn load_with_mode(
        model_path: &str,
        num_experts: usize,
        top_k: usize,
        routing_mode: RoutingMode,
    ) -> Result<Self> {
        let top_k = top_k.max(1).min(num_experts);

        if model_path.is_empty() {
            return Ok(Self {
                model_path: String::new(),
                num_experts,
                top_k,
                loaded: false,
                metadata: OlmoeMetadata {
                    hidden_size: OLMOE_HIDDEN,
                    num_layers: OLMOE_NUM_LAYERS,
                    num_experts: OLMOE_NUM_EXPERTS,
                    quantization: "stub".into(),
                },
                routing_mode,
                expert_membranes: vec![0.0; num_experts],
                hidden_membranes: vec![0.0; EMBEDDING_DIM],
                threshold: 0.75,
                decay: 0.91,
                checkpoint: None,
            });
        }

        let (metadata, checkpoint) = Self::probe_and_map(model_path)?;
        if num_experts > metadata.num_experts {
            return Err(HybridError::InvalidConfig(format!(
                "num_experts ({num_experts}) exceeds checkpoint expert_count ({})",
                metadata.num_experts
            )));
        }

        Ok(Self {
            model_path: model_path.to_owned(),
            num_experts,
            top_k,
            loaded: true,
            metadata,
            routing_mode,
            expert_membranes: vec![0.0; num_experts],
            hidden_membranes: vec![0.0; EMBEDDING_DIM],
            threshold: 0.75,
            decay: 0.91,
            checkpoint: Some(checkpoint),
        })
    }

    pub fn forward(&mut self, embedding: &[f32]) -> Result<OlmoeOutput> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(HybridError::InputLengthMismatch {
                expected: EMBEDDING_DIM,
                got: embedding.len(),
            });
        }

        match self.routing_mode {
            RoutingMode::StubUniform => Ok(self.stub_output()),
            RoutingMode::DenseSim => self.simulate_moe_routing(embedding),
            RoutingMode::SpikingSim => self.spiking_moe_routing(embedding),
        }
    }

    pub fn extract_token_embedding(&mut self, token_id: usize) -> Result<Vec<f32>> {
        let path = self.model_path.clone();
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;

        extract_token_embedding_from_checkpoint(checkpoint, &path, token_id)
    }

    pub(crate) fn registered_gpu_synapse_weights(&mut self, tensor_name: &str) -> Result<&[u16]> {
        let path = self.model_path.clone();
        let checkpoint = self
            .checkpoint
            .as_mut()
            .ok_or_else(|| HybridError::ModelLoad {
                path: path.clone(),
                reason: "checkpoint not loaded".into(),
            })?;
        checkpoint.registered_f16_tensor(tensor_name, &path)
    }

    fn probe_and_map(path: &str) -> Result<(OlmoeMetadata, MappedOlmoeCheckpoint)> {
        probe_and_map_checkpoint(path)
    }

    fn simulate_moe_routing(&self, embedding: &[f32]) -> Result<OlmoeOutput> {
        let gate_scores = self.compute_gate_scores(embedding)?;
        let expert_weights = softmax(&gate_scores);
        let selected_experts = top_k_indices(&expert_weights, self.top_k);
        let selected_mass: f32 = selected_experts
            .iter()
            .map(|&idx| expert_weights[idx])
            .sum();
        let hidden: Vec<f32> = embedding.iter().map(|&v| v * selected_mass).collect();

        Ok(OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn spiking_moe_routing(&mut self, embedding: &[f32]) -> Result<OlmoeOutput> {
        let gate_scores = self.compute_gate_scores(embedding)?;
        let n = self.num_experts;
        let mut membrane_scores = Vec::with_capacity(n);
        let mut expert_spikes = vec![0.0f32; n];

        for expert_id in 0..n {
            self.expert_membranes[expert_id] =
                self.expert_membranes[expert_id] * self.decay + gate_scores[expert_id] * 0.18;

            let spike = if self.expert_membranes[expert_id] > self.threshold {
                self.expert_membranes[expert_id] -= self.threshold;
                1.0
            } else if self.expert_membranes[expert_id] < -self.threshold {
                self.expert_membranes[expert_id] += self.threshold;
                -1.0
            } else {
                0.0
            };

            expert_spikes[expert_id] = spike;
            membrane_scores.push(self.expert_membranes[expert_id] + spike * self.threshold);
        }

        let expert_weights = softmax(&membrane_scores);
        let selected_experts = top_k_indices(&expert_weights, self.top_k);

        let active_mass: f32 = selected_experts
            .iter()
            .map(|&expert_id| expert_spikes[expert_id] * expert_weights[expert_id])
            .sum();

        let mut hidden = vec![0.0f32; EMBEDDING_DIM];
        for (j, h) in hidden.iter_mut().enumerate() {
            let input = embedding[j] * active_mass;
            self.hidden_membranes[j] = self.hidden_membranes[j] * self.decay + input;

            let spike = if self.hidden_membranes[j] > self.threshold {
                self.hidden_membranes[j] -= self.threshold;
                1.0
            } else if self.hidden_membranes[j] < -self.threshold {
                self.hidden_membranes[j] += self.threshold;
                -1.0
            } else {
                0.0
            };

            *h = spike * 0.3;
        }

        Ok(OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        })
    }

    fn compute_gate_scores(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if let Some(checkpoint) = &self.checkpoint {
            return checkpoint_gate_scores(
                checkpoint,
                &self.model_path,
                self.num_experts,
                embedding,
            );
        }

        Ok(synthetic_gate_scores(self.num_experts, embedding))
    }

    fn stub_output(&self) -> OlmoeOutput {
        let n = self.num_experts;
        let expert_weights = vec![1.0 / n as f32; n];
        let selected_experts = (0..self.top_k).collect();
        let hidden = vec![0.0f32; EMBEDDING_DIM];
        OlmoeOutput {
            expert_weights,
            selected_experts,
            hidden,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    pub fn reset_state(&mut self) {
        self.expert_membranes.fill(0.0);
        self.hidden_membranes.fill(0.0);
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn quantization(&self) -> &str {
        &self.metadata.quantization
    }

    pub fn hidden_size(&self) -> usize {
        self.metadata.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.metadata.num_layers
    }

    pub fn checkpoint_num_experts(&self) -> usize {
        self.metadata.num_experts
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn routing_mode(&self) -> RoutingMode {
        self.routing_mode
    }

    #[cfg(test)]
    pub(crate) fn has_state_activity(&self) -> bool {
        self.expert_membranes.iter().any(|&v| v != 0.0)
            || self.hidden_membranes.iter().any(|&v| v != 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuAccelerator;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_temp_file(bytes: &[u8], label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "corinth_canal_{label}_{}.gguf",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(out: &mut Vec<u8>, value: &str) {
        push_u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }

    fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
        push_string(out, key);
        push_u32(out, GGUF_VALUE_TYPE_UINT32);
        push_u32(out, value);
    }

    fn build_test_gguf(tensors: Vec<(&str, Vec<usize>, u32, Vec<u8>)>, alignment: u32) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&GGUF_MAGIC);
        push_u32(&mut out, GGUF_VERSION);
        push_u64(&mut out, tensors.len() as u64);
        push_u64(&mut out, 3);
        push_kv_u32(&mut out, "general.alignment", alignment);
        push_kv_u32(&mut out, "general.file_type", 1);
        push_kv_u32(&mut out, "olmoe.expert_count", 64);

        let mut data_offset = 0usize;
        let mut tensor_payloads = Vec::new();
        for (name, dims, ggml_type, payload) in tensors {
            push_string(&mut out, name);
            push_u32(&mut out, dims.len() as u32);
            for dim in &dims {
                push_u64(&mut out, *dim as u64);
            }
            push_u32(&mut out, ggml_type);
            push_u64(&mut out, data_offset as u64);
            data_offset += payload.len();
            tensor_payloads.push(payload);
        }

        while out.len() % alignment as usize != 0 {
            out.push(0);
        }
        for payload in tensor_payloads {
            out.extend_from_slice(&payload);
        }

        out
    }

    fn build_real_size_checkpoint(gate_payload: Vec<u8>) -> Vec<u8> {
        let attn_q_payload = vec![0u8; OLMOE_HIDDEN * OLMOE_HIDDEN * 2];
        build_test_gguf(
            vec![
                (
                    ROUTING_TENSOR_NAME,
                    vec![EMBEDDING_DIM, OLMOE_NUM_EXPERTS],
                    GGML_TYPE_F32,
                    gate_payload,
                ),
                (
                    DEFAULT_GPU_SYNAPSE_TENSOR_NAME,
                    vec![OLMOE_HIDDEN, OLMOE_HIDDEN],
                    GGML_TYPE_F16,
                    attn_q_payload,
                ),
            ],
            32,
        )
    }

    fn stub() -> OlmoeRouter {
        OlmoeRouter::load_with_mode("", 8, 1, RoutingMode::StubUniform)
            .expect("stub load should succeed")
    }

    fn dense_sim_stub() -> OlmoeRouter {
        OlmoeRouter::load_with_mode("", 8, 2, RoutingMode::DenseSim)
            .expect("dense sim stub load should succeed")
    }

    fn spiking_sim_stub() -> OlmoeRouter {
        OlmoeRouter::load_with_mode("", 8, 2, RoutingMode::SpikingSim)
            .expect("spiking sim stub load should succeed")
    }

    #[test]
    fn test_stub_mode_loads() {
        let model = stub();
        assert!(!model.is_loaded());
        assert_eq!(model.quantization(), "stub");
    }

    #[test]
    fn test_stub_forward_shape() {
        let mut model = stub();
        let embedding = vec![0.0f32; EMBEDDING_DIM];
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.expert_weights.len(), 8);
        assert_eq!(out.selected_experts.len(), 1);
        assert_eq!(out.hidden.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_stub_forward_uniform_weights() {
        let mut model = stub();
        let embedding = vec![0.1f32; EMBEDDING_DIM];
        let out = model.forward(&embedding).unwrap();
        for w in &out.expert_weights {
            assert!((*w - 0.125).abs() < 1e-5, "expected uniform 1/8, got {w}");
        }
    }

    #[test]
    fn test_input_length_mismatch() {
        let mut model = stub();
        let bad_embedding = vec![0.0f32; 64];
        assert!(model.forward(&bad_embedding).is_err());
    }

    #[test]
    fn test_dense_sim_in_stub_mode_has_valid_routing() {
        let mut model = dense_sim_stub();
        let embedding: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|i| (i as f32 / EMBEDDING_DIM as f32) * 0.1)
            .collect();
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.selected_experts.len(), 2);
        let weight_sum: f32 = out.expert_weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1e-5,
            "expert weights must sum to 1, got {weight_sum}"
        );
        assert_eq!(out.hidden.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_spiking_sim_persists_state_and_can_fire() {
        let mut model = spiking_sim_stub();
        let embedding = vec![1.0f32; EMBEDDING_DIM];

        let first = model.forward(&embedding).unwrap();
        assert!(model.expert_membranes.iter().any(|&v| v != 0.0));

        let mut fired = first.hidden.iter().any(|&v| v != 0.0);
        for _ in 0..32 {
            let out = model.forward(&embedding).unwrap();
            if out.hidden.iter().any(|&v| v != 0.0) {
                fired = true;
                break;
            }
        }

        assert!(
            fired,
            "spiking sim should eventually emit ternary hidden events"
        );
    }

    #[test]
    fn test_spiking_sim_reset_clears_state() {
        let mut model = spiking_sim_stub();
        let embedding = vec![1.0f32; EMBEDDING_DIM];

        let _ = model.forward(&embedding).unwrap();
        assert!(model.expert_membranes.iter().any(|&v| v != 0.0));

        model.reset_state();

        assert!(model.expert_membranes.iter().all(|&v| v == 0.0));
        assert!(model.hidden_membranes.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_parse_checkpoint_layout_preserves_tensor_offsets() {
        let bytes = build_test_gguf(
            vec![("demo.weight", vec![2, 2], GGML_TYPE_F32, vec![0u8; 16])],
            64,
        );
        let parsed = checkpoint::parse_checkpoint_layout(&bytes, "test.gguf").unwrap();
        let tensor = parsed.tensors.get("demo.weight").unwrap();
        assert_eq!(tensor.relative_offset, 0);
        assert_eq!(tensor.absolute_offset % 64, 0);
        assert_eq!(tensor.n_elements, 4);
    }

    #[test]
    fn test_probe_and_map_rejects_missing_routing_tensor() {
        let bytes = build_test_gguf(
            vec![(
                DEFAULT_GPU_SYNAPSE_TENSOR_NAME,
                vec![OLMOE_HIDDEN, OLMOE_HIDDEN],
                GGML_TYPE_F16,
                vec![0u8; OLMOE_HIDDEN * OLMOE_HIDDEN * 2],
            )],
            32,
        );
        let path = write_temp_file(&bytes, "missing-routing");
        let err = OlmoeRouter::probe_and_map(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, HybridError::MissingTensor { .. }));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_probe_and_map_rejects_wrong_synapse_type() {
        let gate_payload = vec![0u8; EMBEDDING_DIM * OLMOE_NUM_EXPERTS * 4];
        let bytes = build_test_gguf(
            vec![
                (
                    ROUTING_TENSOR_NAME,
                    vec![EMBEDDING_DIM, OLMOE_NUM_EXPERTS],
                    GGML_TYPE_F32,
                    gate_payload,
                ),
                (
                    DEFAULT_GPU_SYNAPSE_TENSOR_NAME,
                    vec![OLMOE_HIDDEN, OLMOE_HIDDEN],
                    GGML_TYPE_F32,
                    vec![0u8; OLMOE_HIDDEN * OLMOE_HIDDEN * 4],
                ),
            ],
            32,
        );
        let path = write_temp_file(&bytes, "wrong-type");
        let err = OlmoeRouter::probe_and_map(path.to_str().unwrap()).unwrap_err();
        assert!(matches!(err, HybridError::UnsupportedFormat(_)));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_dense_sim_uses_real_gate_weights() {
        let mut gate = vec![0.0f32; EMBEDDING_DIM * OLMOE_NUM_EXPERTS];
        for (expert, gate_value) in gate.iter_mut().take(OLMOE_NUM_EXPERTS).enumerate() {
            *gate_value = if expert == 0 { 8.0 } else { -8.0 };
        }
        let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
        let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "dense-real");

        let mut model =
            OlmoeRouter::load_with_mode(path.to_str().unwrap(), 8, 2, RoutingMode::DenseSim)
                .unwrap();
        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        embedding[0] = 1.0;
        let out = model.forward(&embedding).unwrap();
        assert_eq!(out.selected_experts[0], 0);
        let weight_sum: f32 = out.expert_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_spiking_real_gate_path_accumulates_state() {
        let mut gate = vec![0.0f32; EMBEDDING_DIM * OLMOE_NUM_EXPERTS];
        for (expert, gate_value) in gate.iter_mut().take(OLMOE_NUM_EXPERTS).enumerate() {
            *gate_value = if expert == 0 { 8.0 } else { -8.0 };
        }
        let gate_bytes: Vec<u8> = gate.iter().flat_map(|value| value.to_le_bytes()).collect();
        let path = write_temp_file(&build_real_size_checkpoint(gate_bytes), "spiking-real");

        let mut model =
            OlmoeRouter::load_with_mode(path.to_str().unwrap(), 8, 2, RoutingMode::SpikingSim)
                .unwrap();
        let mut embedding = vec![0.0f32; EMBEDDING_DIM];
        embedding[0] = 1.0;
        for _ in 0..8 {
            let _ = model.forward(&embedding).unwrap();
        }
        assert!(model.has_state_activity());
        model.reset_state();
        assert!(!model.has_state_activity());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_removed_old_weight_symbol() {
        let source = std::fs::read_to_string(file!()).unwrap();
        let pattern = ["pseudo", "_weight"].concat();
        assert!(!source.contains(&pattern));
    }

    #[test]
    fn test_real_checkpoint_probe_via_env() {
        let Some(path) = std::env::var("GGUF_CHECKPOINT_PATH").ok() else {
            return;
        };

        let (metadata, checkpoint) = OlmoeRouter::probe_and_map(&path).unwrap();
        assert_eq!(metadata.hidden_size, 2048);
        assert_eq!(metadata.num_experts, 64);
        let routing = checkpoint.tensor_info(ROUTING_TENSOR_NAME, &path).unwrap();
        assert_eq!(routing.dims, vec![2048, 64]);
        let synapse = checkpoint
            .tensor_info(DEFAULT_GPU_SYNAPSE_TENSOR_NAME, &path)
            .unwrap();
        assert_eq!(synapse.dims, vec![2048, 2048]);
    }

    #[test]
    fn test_registered_gpu_upload_via_env() {
        let Some(path) = std::env::var("GGUF_CHECKPOINT_PATH").ok() else {
            return;
        };
        if !crate::gpu::GpuContext::is_available() {
            return;
        }

        let mut accelerator = GpuAccelerator::new();
        if !accelerator.is_ready() {
            return;
        }

        let mut model = OlmoeRouter::load_with_mode(&path, 8, 1, RoutingMode::DenseSim).unwrap();
        accelerator.ensure_temporal_state(OLMOE_HIDDEN).unwrap();
        let weights = model
            .registered_gpu_synapse_weights(DEFAULT_GPU_SYNAPSE_TENSOR_NAME)
            .unwrap();
        accelerator
            .load_synapse_weights_f16_registered("env::blk.0.attn_q.weight", weights)
            .unwrap();
        assert_eq!(
            accelerator.synapse_signature(),
            Some("env::blk.0.attn_q.weight")
        );
    }
}
