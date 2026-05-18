//! Safetensors checkpoint inspection and deterministic manifest generation.
//!
//! This module reads only Safetensors headers and optional Hugging Face shard
//! index metadata. It does not read tensor payload bytes into memory and does
//! not perform activation tracing or router math.

use crate::error::{HybridError, Result};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const SAFETENSORS_EXTENSION: &str = "safetensors";
const MAX_HEADER_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsManifest {
    pub manifest_version: u32,
    pub format: &'static str,
    pub checkpoint: SafetensorsCheckpointSource,
    pub tensors: Vec<SafetensorsTensorRecord>,
    pub candidates: SafetensorsCandidateSummary,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsCheckpointSource {
    pub input_kind: String,
    pub index_file: Option<String>,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsTensorRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_size: usize,
    pub source_shard: String,
    pub data_offsets: [usize; 2],
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SafetensorsCandidateSummary {
    pub router_tensors: Vec<String>,
    pub expert_tensors: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: BTreeMap<String, Value>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug)]
struct ShardInspection {
    metadata: BTreeMap<String, String>,
    tensors: Vec<SafetensorsTensorRecord>,
}

/// Inspect a Safetensors file, sharded Safetensors index, or directory of
/// Safetensors shards and return a deterministic JSON-serializable manifest.
pub fn inspect_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsManifest> {
    let input = path.as_ref();
    let metadata = fs::metadata(input).map_err(|e| model_load(input, e.to_string()))?;

    if metadata.is_dir() {
        return inspect_directory(input);
    }

    if is_safetensors_index(input) {
        let root = input.parent().unwrap_or_else(|| Path::new("."));
        return inspect_index(root, input);
    }

    if input.extension().and_then(|ext| ext.to_str()) == Some(SAFETENSORS_EXTENSION) {
        return inspect_single_file(input);
    }

    Err(HybridError::UnsupportedFormat(format!(
        "expected .safetensors file, .safetensors.index.json file, or directory, got '{}'",
        input.display()
    )))
}

/// Write a deterministic pretty-printed Safetensors manifest to `output_path`.
pub fn write_safetensors_manifest(
    checkpoint_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<SafetensorsManifest> {
    let manifest = inspect_safetensors_checkpoint(checkpoint_path)?;
    let json = serde_json::to_string_pretty(&manifest).map_err(|e| HybridError::ModelLoad {
        path: output_path.as_ref().display().to_string(),
        reason: format!("serialize Safetensors manifest: {e}"),
    })?;
    fs::write(output_path.as_ref(), json).map_err(|e| HybridError::ModelLoad {
        path: output_path.as_ref().display().to_string(),
        reason: e.to_string(),
    })?;
    Ok(manifest)
}

fn inspect_single_file(path: &Path) -> Result<SafetensorsManifest> {
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    let shard = inspect_shard(path, root)?;
    Ok(build_manifest(
        "single_file",
        None,
        vec![path.to_path_buf()],
        shard.metadata,
        shard.tensors,
    ))
}

fn inspect_directory(root: &Path) -> Result<SafetensorsManifest> {
    if let Some(index_path) = find_index_file(root)? {
        return inspect_index(root, &index_path);
    }

    let shards = list_safetensors_files(root)?;
    if shards.is_empty() {
        return Err(HybridError::UnsupportedFormat(format!(
            "no .safetensors files found in '{}'",
            root.display()
        )));
    }
    inspect_shards("directory", root, None, shards)
}

fn inspect_index(root: &Path, index_path: &Path) -> Result<SafetensorsManifest> {
    let raw = read_index(index_path)?;
    let shards = raw
        .weight_map
        .values()
        .map(|relative| index_shard_path(root, index_path, relative))
        .collect::<Result<BTreeSet<_>>>()?
        .into_iter()
        .collect::<Vec<_>>();

    let mut metadata = stringify_metadata(raw.metadata);
    metadata.insert(
        "index_tensor_count".into(),
        raw.weight_map.len().to_string(),
    );
    let index_file = Some(relative_path(index_path, root));
    inspect_shards("hf_index", root, index_file, shards).map(|mut manifest| {
        manifest.checkpoint.metadata.extend(metadata);
        manifest
    })
}

fn inspect_shards(
    input_kind: &str,
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
) -> Result<SafetensorsManifest> {
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for shard_path in &shards {
        let shard = inspect_shard(shard_path, root)?;
        metadata.extend(shard.metadata);
        tensors.extend(shard.tensors);
    }

    Ok(build_manifest(
        input_kind, index_file, shards, metadata, tensors,
    ))
}

fn build_manifest(
    input_kind: &str,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    metadata: BTreeMap<String, String>,
    mut tensors: Vec<SafetensorsTensorRecord>,
) -> SafetensorsManifest {
    tensors.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then(left.source_shard.cmp(&right.source_shard))
    });
    let router_tensors = tensors
        .iter()
        .filter(|tensor| {
            tensor
                .labels
                .iter()
                .any(|label| label == "moe_router_candidate")
        })
        .map(|tensor| tensor.name.clone())
        .collect();
    let expert_tensors = tensors
        .iter()
        .filter(|tensor| {
            tensor
                .labels
                .iter()
                .any(|label| label == "moe_expert_candidate")
        })
        .map(|tensor| tensor.name.clone())
        .collect();

    SafetensorsManifest {
        manifest_version: 1,
        format: "safetensors",
        checkpoint: SafetensorsCheckpointSource {
            input_kind: input_kind.to_string(),
            index_file,
            shard_count: shards.len(),
            tensor_count: tensors.len(),
            metadata,
        },
        tensors,
        candidates: SafetensorsCandidateSummary {
            router_tensors,
            expert_tensors,
        },
    }
}

fn inspect_shard(path: &Path, root: &Path) -> Result<ShardInspection> {
    let mut file = File::open(path).map_err(|e| model_load(path, e.to_string()))?;
    let file_len = file
        .metadata()
        .map_err(|e| model_load(path, e.to_string()))?
        .len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header length: {e}")))?;
    let header_len_u64 = u64::from_le_bytes(len_bytes);
    let header_len = usize::try_from(header_len_u64).map_err(|_| {
        model_load(
            path,
            format!("Safetensors header length {header_len_u64} does not fit in usize"),
        )
    })?;
    if header_len > MAX_HEADER_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
        ));
    }
    if 8u64
        .checked_add(header_len_u64)
        .is_none_or(|end| end > file_len)
    {
        return Err(model_load(
            path,
            "Safetensors header extends beyond file".into(),
        ));
    }

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header: {e}")))?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| model_load(path, format!("parse Safetensors header JSON: {e}")))?;
    parse_header(path, root, file_len, header_len_u64, header)
}

fn parse_header(
    path: &Path,
    root: &Path,
    file_len: u64,
    header_len: u64,
    header: Value,
) -> Result<ShardInspection> {
    let object = header
        .as_object()
        .ok_or_else(|| model_load(path, "Safetensors header must be a JSON object".to_string()))?;
    let data_len = file_len
        .checked_sub(8)
        .and_then(|len| len.checked_sub(header_len))
        .ok_or_else(|| model_load(path, "Safetensors data range underflow".into()))?;
    let source_shard = relative_path(path, root);
    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();

    for (name, value) in object {
        if name == "__metadata__" {
            metadata.extend(stringify_value_object(value));
            continue;
        }

        let tensor = value.as_object().ok_or_else(|| {
            model_load(path, format!("tensor '{name}' metadata must be an object"))
        })?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing dtype")))?
            .to_string();
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing shape")))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| model_load(path, format!("tensor '{name}' is missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets must have length 2"),
            ));
        }
        let start = offsets[0]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid start offset")))?;
        let end = offsets[1]
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| model_load(path, format!("tensor '{name}' has invalid end offset")))?;
        if start > end {
            return Err(model_load(
                path,
                format!("tensor '{name}' data_offsets are reversed"),
            ));
        }
        if u64::try_from(end).map_or(true, |end| end > data_len) {
            return Err(model_load(
                path,
                format!("tensor '{name}' extends beyond Safetensors data section"),
            ));
        }

        tensors.push(SafetensorsTensorRecord {
            name: name.clone(),
            dtype,
            shape: shape.clone(),
            byte_size: end - start,
            source_shard: source_shard.clone(),
            data_offsets: [start, end],
            labels: classify_tensor(name, &shape),
        });
    }

    Ok(ShardInspection { metadata, tensors })
}

fn classify_tensor(name: &str, shape: &[usize]) -> Vec<String> {
    let lower = name.to_ascii_lowercase();
    let mut labels = Vec::new();

    let router_name = lower.contains("router")
        || lower.contains("gate_inp")
        || lower.ends_with(".gate.weight")
        || lower.ends_with("/gate/weight");
    let router_shape = shape.len() == 2
        && shape
            .iter()
            .copied()
            .min()
            .is_some_and(|v| (2..=512).contains(&v))
        && shape.iter().copied().max().is_some_and(|v| v >= 512);
    if router_name || router_shape {
        labels.push("moe_router_candidate".to_string());
    }

    let expert_name = lower.contains("expert")
        || lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
        || lower.contains(".w1.")
        || lower.contains(".w2.")
        || lower.contains(".w3.");
    if expert_name {
        labels.push("moe_expert_candidate".to_string());
    }

    labels
}

fn read_index(path: &Path) -> Result<RawIndex> {
    let bytes = fs::read(path).map_err(|e| model_load(path, e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| model_load(path, format!("parse index JSON: {e}")))
}

fn find_index_file(root: &Path) -> Result<Option<PathBuf>> {
    let mut candidates = fs::read_dir(root)
        .map_err(|e| model_load(root, e.to_string()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| is_safetensors_index(path))
        .collect::<Vec<_>>();
    candidates.sort();
    Ok(candidates.into_iter().next())
}

fn list_safetensors_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = fs::read_dir(root)
        .map_err(|e| model_load(root, e.to_string()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(SAFETENSORS_EXTENSION))
        .collect::<Vec<_>>();
    shards.sort();
    Ok(shards)
}

fn is_safetensors_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
}

fn index_shard_path(root: &Path, index_path: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    let escapes_root = relative_path.is_absolute()
        || relative_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        });
    if escapes_root {
        return Err(model_load(
            index_path,
            format!("index shard path '{relative}' must stay within the checkpoint directory"),
        ));
    }
    Ok(root.join(relative_path))
}

fn stringify_metadata(metadata: BTreeMap<String, Value>) -> BTreeMap<String, String> {
    metadata
        .into_iter()
        .map(|(key, value)| (key, stringify_json_value(&value)))
        .collect()
}

fn stringify_value_object(value: &Value) -> BTreeMap<String, String> {
    value
        .as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), stringify_json_value(value)))
                .collect()
        })
        .unwrap_or_default()
}

fn stringify_json_value(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn relative_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn model_load(path: &Path, reason: String) -> HybridError {
    HybridError::ModelLoad {
        path: path.display().to_string(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "corinth-safetensors-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_safetensors(path: &Path, header: &str, data_len: usize) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&vec![0u8; data_len]).unwrap();
    }

    #[test]
    fn parses_single_file_manifest_and_labels_candidates() {
        let dir = temp_dir("single");
        let path = dir.join("model.safetensors");
        write_safetensors(
            &path,
            r#"{
                "__metadata__": {"format": "pt"},
                "model.layers.0.mlp.gate.weight": {"dtype": "F32", "shape": [64, 2048], "data_offsets": [0, 524288]},
                "model.layers.0.mlp.experts.0.w1.weight": {"dtype": "F16", "shape": [2048, 4096], "data_offsets": [524288, 17301504]}
            }"#,
            17_301_504,
        );

        let manifest = inspect_safetensors_checkpoint(&path).unwrap();
        assert_eq!(manifest.checkpoint.input_kind, "single_file");
        assert_eq!(manifest.checkpoint.shard_count, 1);
        assert_eq!(manifest.checkpoint.tensor_count, 2);
        assert_eq!(manifest.checkpoint.metadata.get("format").unwrap(), "pt");
        assert_eq!(
            manifest.tensors[0].name,
            "model.layers.0.mlp.experts.0.w1.weight"
        );
        assert_eq!(manifest.tensors[0].source_shard, "model.safetensors");
        assert_eq!(manifest.candidates.router_tensors.len(), 1);
        assert_eq!(manifest.candidates.expert_tensors.len(), 1);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_hugging_face_index_and_orders_deterministically() {
        let dir = temp_dir("index");
        let shard_a = dir.join("model-00001-of-00002.safetensors");
        let shard_b = dir.join("model-00002-of-00002.safetensors");
        write_safetensors(
            &shard_b,
            r#"{"z.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]}}"#,
            8,
        );
        write_safetensors(
            &shard_a,
            r#"{"a.router.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]}}"#,
            65_536,
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "metadata": {"total_size": 65544},
                "weight_map": {
                    "z.weight": "model-00002-of-00002.safetensors",
                    "a.router.weight": "model-00001-of-00002.safetensors"
                }
            }"#,
        )
        .unwrap();

        let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
        assert_eq!(manifest.checkpoint.input_kind, "hf_index");
        assert_eq!(
            manifest.checkpoint.index_file.as_deref(),
            Some("model.safetensors.index.json")
        );
        assert_eq!(manifest.checkpoint.shard_count, 2);
        assert_eq!(
            manifest.checkpoint.metadata.get("total_size").unwrap(),
            "65544"
        );
        assert_eq!(manifest.tensors[0].name, "a.router.weight");
        assert_eq!(manifest.tensors[1].name, "z.weight");
        assert_eq!(manifest.candidates.router_tensors, vec!["a.router.weight"]);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_tensor_offsets_beyond_data_section() {
        let dir = temp_dir("bounds");
        let path = dir.join("bad.safetensors");
        write_safetensors(
            &path,
            r#"{"bad.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 16]}}"#,
            8,
        );

        let err = inspect_safetensors_checkpoint(&path).unwrap_err();
        assert!(
            err.to_string()
                .contains("extends beyond Safetensors data section")
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_index_shard_paths_that_escape_checkpoint_directory() {
        let dir = temp_dir("escape");
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "weight_map": {
                    "a.weight": "../outside.safetensors"
                }
            }"#,
        )
        .unwrap();

        let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
        assert!(
            err.to_string()
                .contains("must stay within the checkpoint directory")
        );

        fs::remove_dir_all(dir).unwrap();
    }
}
