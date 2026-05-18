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
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;

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
    pub labels: Vec<&'static str>,
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
    let checkpoint_path = checkpoint_path.as_ref();
    let output_path = output_path.as_ref();
    let manifest = inspect_safetensors_checkpoint(checkpoint_path)?;
    reject_output_checkpoint_conflict(checkpoint_path, output_path, &manifest)?;
    let json = serde_json::to_string_pretty(&manifest).map_err(|e| HybridError::ModelLoad {
        path: output_path.display().to_string(),
        reason: format!("serialize Safetensors manifest: {e}"),
    })?;
    fs::write(output_path, json).map_err(|e| HybridError::ModelLoad {
        path: output_path.display().to_string(),
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
    let index_tensor_count = raw.weight_map.len();
    let mut expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for (tensor_name, relative) in raw.weight_map {
        let shard_path = index_shard_path(root, index_path, &relative)?;
        expected_by_shard
            .entry(shard_path)
            .or_default()
            .insert(tensor_name);
    }
    let shards = expected_by_shard.keys().cloned().collect::<Vec<_>>();

    let mut metadata = stringify_metadata("index", raw.metadata);
    metadata.insert("index_tensor_count".into(), index_tensor_count.to_string());
    let index_file = Some(relative_path(index_path, root));
    inspect_index_shards(root, index_file, shards, expected_by_shard, metadata)
}

fn inspect_index_shards(
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>>,
    mut metadata: BTreeMap<String, String>,
) -> Result<SafetensorsManifest> {
    let mut tensors = Vec::new();
    for shard_path in &shards {
        let expected = expected_by_shard.get(shard_path).ok_or_else(|| {
            model_load(
                shard_path,
                "internal error: index shard has no expected tensor set".into(),
            )
        })?;
        let shard = inspect_shard(shard_path, root)?;
        merge_shard_metadata(
            &mut metadata,
            &relative_path(shard_path, root),
            shard.metadata,
        );

        let found = shard
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<BTreeSet<_>>();
        if let Some(missing) = expected.difference(&found).next() {
            return Err(model_load(
                shard_path,
                format!(
                    "index maps tensor '{missing}' to this shard, but the shard header does not contain it"
                ),
            ));
        }

        tensors.extend(
            shard
                .tensors
                .into_iter()
                .filter(|tensor| expected.contains(&tensor.name)),
        );
    }

    Ok(build_manifest(
        "hf_index", index_file, shards, metadata, tensors,
    ))
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
        merge_shard_metadata(
            &mut metadata,
            &relative_path(shard_path, root),
            shard.metadata,
        );
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
        .filter(|tensor| tensor.labels.contains(&"moe_router_candidate"))
        .map(|tensor| tensor.name.clone())
        .collect();
    let expert_tensors = tensors
        .iter()
        .filter(|tensor| tensor.labels.contains(&"moe_expert_candidate"))
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
        if end as u64 > data_len {
            return Err(model_load(
                path,
                format!("tensor '{name}' extends beyond Safetensors data section"),
            ));
        }
        let byte_size = end - start;
        match expected_tensor_byte_size(&dtype, &shape, path, name)? {
            Some(expected) if expected != byte_size => {
                return Err(model_load(
                    path,
                    format!(
                        "tensor '{name}' byte size mismatch: shape/dtype imply {expected} bytes, data_offsets span {byte_size} bytes"
                    ),
                ));
            }
            _ => {}
        }

        tensors.push(SafetensorsTensorRecord {
            name: name.clone(),
            dtype,
            shape: shape.clone(),
            byte_size,
            source_shard: source_shard.clone(),
            data_offsets: [start, end],
            labels: classify_tensor(name, &shape),
        });
    }

    Ok(ShardInspection { metadata, tensors })
}

fn classify_tensor(name: &str, shape: &[usize]) -> Vec<&'static str> {
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
    if router_name {
        labels.push("moe_router_candidate");
    } else if router_shape {
        labels.push("possible_moe_router_shape");
    }

    let has_expert_context = lower.contains("experts")
        || lower.contains(".expert")
        || lower.contains("/expert")
        || lower.contains("block_sparse_moe.experts");
    let expert_weight_name = lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
        || lower.contains(".w1.")
        || lower.contains(".w2.")
        || lower.contains(".w3.");
    let expert_name = has_expert_context && expert_weight_name;
    if expert_name {
        labels.push("moe_expert_candidate");
    }

    labels
}

fn read_index(path: &Path) -> Result<RawIndex> {
    let len = fs::metadata(path)
        .map_err(|e| model_load(path, e.to_string()))?
        .len();
    if len > MAX_INDEX_BYTES {
        return Err(model_load(
            path,
            format!("Safetensors index is {len} bytes, exceeding limit {MAX_INDEX_BYTES}"),
        ));
    }
    let bytes = fs::read(path).map_err(|e| model_load(path, e.to_string()))?;
    serde_json::from_slice(&bytes).map_err(|e| model_load(path, format!("parse index JSON: {e}")))
}

fn find_index_file(root: &Path) -> Result<Option<PathBuf>> {
    let mut candidates = read_dir_paths(root)?
        .into_iter()
        .filter(|path| is_safetensors_index(path))
        .collect::<Vec<_>>();
    candidates.sort();
    if candidates.len() > 1 {
        let names = candidates
            .iter()
            .map(|path| relative_path(path, root))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(HybridError::UnsupportedFormat(format!(
            "multiple Safetensors index files found in '{}': {names}; pass the intended index file explicitly",
            root.display()
        )));
    }
    Ok(candidates.pop())
}

fn list_safetensors_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = read_dir_paths(root)?
        .into_iter()
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some(SAFETENSORS_EXTENSION))
        .map(|path| validate_path_stays_under_root(root, &path).map(|()| path))
        .collect::<Result<Vec<_>>>()?;
    shards.sort();
    Ok(shards)
}

fn read_dir_paths(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(root).map_err(|e| model_load(root, e.to_string()))? {
        let entry = entry.map_err(|e| model_load(root, format!("read directory entry: {e}")))?;
        paths.push(entry.path());
    }
    Ok(paths)
}

fn expected_tensor_byte_size(
    dtype: &str,
    shape: &[usize],
    path: &Path,
    tensor_name: &str,
) -> Result<Option<usize>> {
    let Some(element_size) = dtype_size_bytes(dtype) else {
        return Ok(None);
    };
    let elements = shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            model_load(
                path,
                format!("tensor '{tensor_name}' shape element count overflow"),
            )
        })
    })?;
    elements
        .checked_mul(element_size)
        .map(Some)
        .ok_or_else(|| model_load(path, format!("tensor '{tensor_name}' byte size overflow")))
}

fn dtype_size_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F64" | "I64" | "U64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F8_E5M2" | "F8_E4M3" | "I8" | "U8" | "BOOL" => Some(1),
        _ => None,
    }
}

fn merge_shard_metadata(
    metadata: &mut BTreeMap<String, String>,
    shard_path: &str,
    shard_metadata: BTreeMap<String, String>,
) {
    for (key, value) in shard_metadata {
        match metadata.get(&key) {
            None => {
                metadata.insert(key, value);
            }
            Some(existing) if existing == &value => {}
            Some(_) => {
                metadata.insert(format!("shard:{shard_path}:{key}"), value);
            }
        }
    }
}

fn reject_output_checkpoint_conflict(
    checkpoint_path: &Path,
    output_path: &Path,
    manifest: &SafetensorsManifest,
) -> Result<()> {
    let output = canonical_existing_or_parent(output_path)?;
    let input_metadata = fs::metadata(checkpoint_path)
        .map_err(|e| model_load(checkpoint_path, format!("stat checkpoint input: {e}")))?;
    let root = if input_metadata.is_dir() {
        checkpoint_path
    } else {
        checkpoint_path.parent().unwrap_or_else(|| Path::new("."))
    };

    let mut checkpoint_files = BTreeSet::new();
    if input_metadata.is_file() {
        checkpoint_files.insert(checkpoint_path.to_path_buf());
    }
    if let Some(index_file) = &manifest.checkpoint.index_file {
        checkpoint_files.insert(root.join(index_file));
    }
    for tensor in &manifest.tensors {
        checkpoint_files.insert(root.join(&tensor.source_shard));
    }

    for checkpoint_file in checkpoint_files {
        if checkpoint_file.exists()
            && fs::canonicalize(&checkpoint_file).map_err(|e| {
                model_load(
                    &checkpoint_file,
                    format!("canonicalize checkpoint file for output conflict check: {e}"),
                )
            })? == output
        {
            return Err(model_load(
                output_path,
                "manifest output path must not overwrite a Safetensors checkpoint or index file"
                    .into(),
            ));
        }
    }

    Ok(())
}

fn canonical_existing_or_parent(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path).map_err(|e| model_load(path, e.to_string()));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        model_load(
            path,
            "manifest output path must include a file name".to_string(),
        )
    })?;
    let parent = fs::canonicalize(parent).map_err(|e| model_load(parent, e.to_string()))?;
    Ok(parent.join(file_name))
}

fn validate_path_stays_under_root(root: &Path, path: &Path) -> Result<()> {
    let canonical_root = fs::canonicalize(root).map_err(|e| model_load(root, e.to_string()))?;
    let canonical_path = fs::canonicalize(path).map_err(|e| model_load(path, e.to_string()))?;
    if !canonical_path.starts_with(&canonical_root) {
        return Err(model_load(
            path,
            "Safetensors shard path must stay within the checkpoint directory".into(),
        ));
    }
    Ok(())
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
    let candidate = root.join(relative_path);
    validate_path_stays_under_root(root, &candidate)?;
    Ok(candidate)
}

fn stringify_metadata(prefix: &str, metadata: BTreeMap<String, Value>) -> BTreeMap<String, String> {
    metadata
        .into_iter()
        .map(|(key, value)| (format!("{prefix}:{key}"), stringify_json_value(&value)))
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
    use std::ops::Deref;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir(PathBuf);

    impl Deref for TestDir {
        type Target = Path;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl AsRef<Path> for TestDir {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(name: &str) -> TestDir {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "corinth-safetensors-{name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        TestDir(path)
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
            manifest
                .checkpoint
                .metadata
                .get("index:total_size")
                .unwrap(),
            "65544"
        );
        assert_eq!(manifest.tensors[0].name, "a.router.weight");
        assert_eq!(manifest.tensors[1].name, "z.weight");
        assert_eq!(manifest.candidates.router_tensors, vec!["a.router.weight"]);
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
    }

    #[cfg(unix)]
    #[test]
    fn rejects_index_shard_paths_that_escape_via_symlink() {
        let dir = temp_dir("symlink");
        let outside = temp_dir("outside");
        let outside_shard = outside.join("outside.safetensors");
        write_safetensors(
            &outside_shard,
            r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );
        std::os::unix::fs::symlink(&outside_shard, dir.join("link.safetensors")).unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "weight_map": {
                    "a.weight": "link.safetensors"
                }
            }"#,
        )
        .unwrap();

        let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
        assert!(
            err.to_string()
                .contains("must stay within the checkpoint directory")
        );
    }

    #[test]
    fn rejects_multiple_directory_indexes() {
        let dir = temp_dir("multiple-indexes");
        fs::write(
            dir.join("a.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .unwrap();
        fs::write(
            dir.join("b.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .unwrap();

        let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
        assert!(err.to_string().contains("multiple Safetensors index files"));
    }

    #[test]
    fn hf_index_filters_extra_tensors_and_requires_mapped_tensors() {
        let dir = temp_dir("index-filter");
        let shard = dir.join("model-00001-of-00001.safetensors");
        write_safetensors(
            &shard,
            r#"{
                "a.router.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [0, 65536]},
                "stale.router.weight": {"dtype": "F32", "shape": [8, 2048], "data_offsets": [65536, 131072]}
            }"#,
            131_072,
        );
        let index = dir.join("model.safetensors.index.json");
        fs::write(
            &index,
            r#"{"weight_map": {"a.router.weight": "model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
        assert_eq!(manifest.checkpoint.tensor_count, 1);
        assert_eq!(manifest.tensors[0].name, "a.router.weight");

        fs::write(
            &index,
            r#"{"weight_map": {"missing.weight": "model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        let err = inspect_safetensors_checkpoint(&dir).unwrap_err();
        assert!(err.to_string().contains("does not contain it"));
    }

    #[test]
    fn rejects_shape_dtype_byte_size_mismatch() {
        let dir = temp_dir("byte-size");
        let path = dir.join("bad.safetensors");
        write_safetensors(
            &path,
            r#"{"bad.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 4]}}"#,
            4,
        );

        let err = inspect_safetensors_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("byte size mismatch"));
    }

    #[test]
    fn rejects_manifest_output_that_overwrites_checkpoint_file() {
        let dir = temp_dir("output-conflict");
        let path = dir.join("model.safetensors");
        write_safetensors(
            &path,
            r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );

        let err = write_safetensors_manifest(&path, &path).unwrap_err();
        assert!(err.to_string().contains("must not overwrite"));
    }

    #[test]
    fn generic_dense_ffn_names_are_not_moe_expert_candidates() {
        let labels = classify_tensor("model.layers.0.mlp.gate_proj.weight", &[4096, 2048]);
        assert!(!labels.contains(&"moe_expert_candidate"));

        let labels = classify_tensor("model.layers.0.block_sparse_moe.gate.weight", &[8, 2048]);
        assert!(labels.contains(&"moe_router_candidate"));
        assert!(!labels.contains(&"moe_expert_candidate"));
    }
}
