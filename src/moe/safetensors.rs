//! Safetensors checkpoint inspection and deterministic manifest generation.
//!
//! This module reads only Safetensors headers and optional Hugging Face shard
//! index metadata. It does not read tensor payload bytes into memory and does
//! not perform activation tracing or router math.

use crate::error::{HybridError, Result};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
#[cfg(windows)]
use std::os::windows::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};

const SAFETENSORS_EXTENSION: &str = "safetensors";
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const INDEX_UNREFERENCED_SHARDS_KEY: &str = "index:unreferenced_shards";
/// Unambiguous boundary between shard relative path and logical metadata key in
/// `shard:*` manifest keys (avoids ambiguity when the logical key contains `:`).
const SHARD_METADATA_KEY_SEP: char = '\u{001e}';

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

struct NoDuplicateValue(Value);

/// Inspect a Safetensors file, sharded Safetensors index, or directory of
/// Safetensors shards and return a deterministic JSON-serializable manifest.
pub fn inspect_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsManifest> {
    let input = path.as_ref();
    let metadata = fs::metadata(input).map_err(|e| model_load(input, e.to_string()))?;

    if metadata.is_dir() {
        return inspect_directory(input);
    }

    if is_safetensors_index(input) {
        let root = parent_or_current(input);
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
    let root = parent_or_current(path);
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
    metadata.remove(INDEX_UNREFERENCED_SHARDS_KEY);
    let indexed_shards = shards.iter().cloned().collect::<BTreeSet<_>>();
    let unreferenced_shards = list_safetensors_files(root)?
        .into_iter()
        .filter(|path| !indexed_shards.contains(path))
        .map(|path| relative_path(&path, root))
        .collect::<Vec<_>>();
    let unreferenced_shards_json = if unreferenced_shards.is_empty() {
        None
    } else {
        Some(serde_json::to_string(&unreferenced_shards).map_err(|e| {
            model_load(
                index_path,
                format!("serialize {INDEX_UNREFERENCED_SHARDS_KEY} list: {e}"),
            )
        })?)
    };
    let index_file = Some(relative_path(index_path, root));
    inspect_index_shards(
        root,
        index_file,
        shards,
        expected_by_shard,
        metadata,
        index_tensor_count,
        unreferenced_shards_json,
    )
}

fn inspect_index_shards(
    root: &Path,
    index_file: Option<String>,
    shards: Vec<PathBuf>,
    expected_by_shard: BTreeMap<PathBuf, BTreeSet<String>>,
    mut metadata: BTreeMap<String, String>,
    index_tensor_count: usize,
    unreferenced_shards_json: Option<String>,
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

    metadata.insert("index_tensor_count".into(), index_tensor_count.to_string());
    if let Some(encoded) = unreferenced_shards_json {
        metadata.insert(INDEX_UNREFERENCED_SHARDS_KEY.into(), encoded);
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
    let header = parse_json_rejecting_duplicate_keys(&header_bytes, path, "Safetensors header")?;
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
            metadata.extend(parse_metadata_object(path, value)?);
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
        let expected = expected_tensor_byte_size(&dtype, &shape, path, name)?;
        if expected != byte_size {
            return Err(model_load(
                path,
                format!(
                    "tensor '{name}' byte size mismatch: shape/dtype imply {expected} bytes, data_offsets span {byte_size} bytes"
                ),
            ));
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

    reject_tensor_data_ranges(path, &tensors, data_len)?;

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

    let has_expert_context =
        lower.contains("experts") || lower.contains(".expert") || lower.contains("/expert");
    let expert_weight_name = lower.contains("gate_proj")
        || lower.contains("up_proj")
        || lower.contains("down_proj")
        || lower.contains(".w1.")
        || lower.contains(".w2.")
        || lower.contains(".w3.");
    let expert_name = has_expert_context && expert_weight_name;
    if expert_name {
        labels.push("moe_expert_candidate");
    } else if router_name {
        labels.push("moe_router_candidate");
    } else if router_shape {
        labels.push("possible_moe_router_shape");
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
    let value = parse_json_rejecting_duplicate_keys(&bytes, path, "Safetensors index")?;
    serde_json::from_value(value).map_err(|e| model_load(path, format!("parse index JSON: {e}")))
}

fn find_index_file(root: &Path) -> Result<Option<PathBuf>> {
    let mut candidates = Vec::new();
    for path in read_dir_paths(root)? {
        if is_safetensors_index(&path) && is_regular_file(&path)? {
            validate_path_stays_under_root(root, &path)?;
            candidates.push(path);
        }
    }
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
    let mut shards = Vec::new();
    for path in read_dir_paths(root)? {
        if path.extension().and_then(|ext| ext.to_str()) != Some(SAFETENSORS_EXTENSION) {
            continue;
        }
        if !is_regular_file(&path)? {
            continue;
        }
        validate_path_stays_under_root(root, &path)?;
        shards.push(path);
    }
    shards.sort();
    Ok(shards)
}

fn is_regular_file(path: &Path) -> Result<bool> {
    fs::metadata(path)
        .map(|metadata| metadata.is_file())
        .map_err(|e| model_load(path, e.to_string()))
}

fn reject_tensor_data_ranges(
    path: &Path,
    tensors: &[SafetensorsTensorRecord],
    data_len: u64,
) -> Result<()> {
    let data_len = usize::try_from(data_len).map_err(|_| {
        model_load(
            path,
            format!("Safetensors data section length {data_len} does not fit in usize"),
        )
    })?;
    let mut ranges = tensors
        .iter()
        .map(|tensor| {
            (
                tensor.data_offsets[0],
                tensor.data_offsets[1],
                tensor.name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|(start, end, name)| (*start, *end, *name));

    let mut expected_start = 0usize;
    let mut previous_name: Option<&str> = None;
    for (start, end, name) in ranges {
        if start < expected_start {
            return Err(model_load(
                path,
                format!(
                    "tensor '{name}' data range overlaps tensor '{}'",
                    previous_name.unwrap_or("<unknown>")
                ),
            ));
        }
        if start > expected_start {
            return Err(model_load(
                path,
                format!("tensor '{name}' data range leaves a gap from {expected_start} to {start}"),
            ));
        }
        expected_start = end;
        previous_name = Some(name);
    }

    if expected_start != data_len {
        return Err(model_load(
            path,
            format!(
                "Safetensors tensor data ranges end at {expected_start}, but data section is {data_len} bytes"
            ),
        ));
    }

    Ok(())
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
) -> Result<usize> {
    let element_size = dtype_size_bytes(dtype).ok_or_else(|| {
        model_load(
            path,
            format!("tensor '{tensor_name}' has unsupported Safetensors dtype '{dtype}'"),
        )
    })?;
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
        .ok_or_else(|| model_load(path, format!("tensor '{tensor_name}' byte size overflow")))
}

fn dtype_size_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "C128" => Some(16),
        "F64" | "I64" | "U64" | "C64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "I8" | "U8" | "BOOL" => Some(1),
        _ => None,
    }
}

fn shard_metadata_namespaced_key(shard_path: &str, key: &str) -> String {
    format!("shard:{shard_path}{SHARD_METADATA_KEY_SEP}{key}")
}

fn namespaced_shard_metadata_logical_key(namespaced: &str) -> Option<&str> {
    let rest = namespaced.strip_prefix("shard:")?;
    if let Some((_, logical_key)) = rest.split_once(SHARD_METADATA_KEY_SEP) {
        return Some(logical_key);
    }
    // Legacy keys used `shard:{path}:{key}` with a single `:` separator; this
    // remains ambiguous when `key` contains `:`, so prefer new keys above.
    rest.rsplit_once(':').map(|(_, logical_key)| logical_key)
}

fn merge_shard_metadata(
    metadata: &mut BTreeMap<String, String>,
    shard_path: &str,
    shard_metadata: BTreeMap<String, String>,
) {
    for (key, value) in shard_metadata {
        let namespaced_key = shard_metadata_namespaced_key(shard_path, &key);
        match metadata.get(&key) {
            None => {
                if !has_shard_metadata_key(metadata, &key) {
                    metadata.insert(key.clone(), value.clone());
                }
                metadata.insert(namespaced_key, value);
            }
            Some(existing) if existing == &value => {
                metadata.insert(namespaced_key, value);
            }
            Some(_) => {
                metadata.remove(&key);
                metadata.insert(namespaced_key, value);
            }
        }
    }
}

fn has_shard_metadata_key(metadata: &BTreeMap<String, String>, key: &str) -> bool {
    metadata
        .keys()
        .any(|existing| namespaced_shard_metadata_logical_key(existing).is_some_and(|k| k == key))
}

fn parse_unreferenced_shards_metadata(encoded: &str) -> Vec<String> {
    if let Ok(paths) = serde_json::from_str::<Vec<String>>(encoded) {
        return paths;
    }
    encoded
        .split(',')
        .filter(|shard| !shard.is_empty())
        .map(str::to_string)
        .collect()
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
        parent_or_current(checkpoint_path)
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
    if let Some(unreferenced_shards) = manifest
        .checkpoint
        .metadata
        .get(INDEX_UNREFERENCED_SHARDS_KEY)
    {
        for shard in parse_unreferenced_shards_metadata(unreferenced_shards) {
            checkpoint_files.insert(root.join(shard));
        }
    }

    for checkpoint_file in checkpoint_files {
        if paths_refer_to_same_file(&checkpoint_file, &output).map_err(|e| {
            model_load(
                &checkpoint_file,
                format!("compare checkpoint/output file identity: {e}"),
            )
        })? {
            return Err(model_load(
                output_path,
                "manifest output path must not overwrite a Safetensors checkpoint or index file"
                    .into(),
            ));
        }
    }

    Ok(())
}

fn paths_refer_to_same_file(left: &Path, right: &Path) -> std::io::Result<bool> {
    if !left.exists() || !right.exists() {
        return Ok(false);
    }

    let left_meta = fs::metadata(left)?;
    let right_meta = fs::metadata(right)?;

    #[cfg(unix)]
    {
        Ok(left_meta.dev() == right_meta.dev() && left_meta.ino() == right_meta.ino())
    }

    #[cfg(windows)]
    {
        Ok(
            left_meta.volume_serial_number() == right_meta.volume_serial_number()
                && left_meta.file_index() == right_meta.file_index(),
        )
    }

    #[cfg(not(any(unix, windows)))]
    {
        Ok(fs::canonicalize(left)? == fs::canonicalize(right)?)
    }
}

fn canonical_existing_or_parent(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return fs::canonicalize(path).map_err(|e| model_load(path, e.to_string()));
    }
    let parent = parent_or_current(path);
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

fn parent_or_current(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn is_safetensors_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".safetensors.index.json"))
}

fn index_shard_path(root: &Path, index_path: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    let mut normalized = PathBuf::new();
    let escapes_root = relative_path.is_absolute()
        || relative_path.components().any(|component| match component {
            Component::Normal(part) => {
                normalized.push(part);
                false
            }
            Component::CurDir => false,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => true,
        });
    if escapes_root {
        return Err(model_load(
            index_path,
            format!("index shard path '{relative}' must stay within the checkpoint directory"),
        ));
    }
    if normalized.as_os_str().is_empty() {
        return Err(model_load(
            index_path,
            format!("index shard path '{relative}' must name a Safetensors shard"),
        ));
    }
    let candidate = root.join(normalized);
    validate_path_stays_under_root(root, &candidate)?;
    Ok(candidate)
}

fn stringify_metadata(prefix: &str, metadata: BTreeMap<String, Value>) -> BTreeMap<String, String> {
    metadata
        .into_iter()
        .map(|(key, value)| (format!("{prefix}:{key}"), stringify_json_value(&value)))
        .collect()
}

fn parse_json_rejecting_duplicate_keys(bytes: &[u8], path: &Path, context: &str) -> Result<Value> {
    serde_json::from_slice::<NoDuplicateValue>(bytes)
        .map(|value| value.0)
        .map_err(|e| model_load(path, format!("parse {context} JSON: {e}")))
}

fn parse_metadata_object(path: &Path, value: &Value) -> Result<BTreeMap<String, String>> {
    let object = value.as_object().ok_or_else(|| {
        model_load(
            path,
            "Safetensors __metadata__ must be a JSON object of strings".into(),
        )
    })?;
    object
        .iter()
        .map(|(key, value)| {
            let value = value.as_str().ok_or_else(|| {
                model_load(
                    path,
                    format!("Safetensors __metadata__ key '{key}' must be a string"),
                )
            })?;
            Ok((key.clone(), value.to_string()))
        })
        .collect()
}

fn stringify_json_value(value: &Value) -> String {
    value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string())
}

impl<'de> Deserialize<'de> for NoDuplicateValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NoDuplicateValueVisitor)
    }
}

struct NoDuplicateValueVisitor;

impl<'de> Visitor<'de> for NoDuplicateValueVisitor {
    type Value = NoDuplicateValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("valid JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        let number = Number::from_f64(value).ok_or_else(|| E::custom("invalid JSON number"))?;
        Ok(NoDuplicateValue(Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(NoDuplicateValue(Value::String(value.to_string())))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::String(value)))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Null))
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(NoDuplicateValue(Value::Null))
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<NoDuplicateValue>()? {
            values.push(value.0);
        }
        Ok(NoDuplicateValue(Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = BTreeSet::new();
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate JSON key '{key}'")));
            }
            let value = map.next_value::<NoDuplicateValue>()?;
            object.insert(key, value.0);
        }
        Ok(NoDuplicateValue(Value::Object(object)))
    }
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
    fn user_index_metadata_cannot_spoof_unreferenced_shards() {
        let dir = temp_dir("reserved-index-metadata");
        write_safetensors(
            &dir.join("model.safetensors"),
            r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "metadata": {"unreferenced_shards": "spoof.safetensors"},
                "weight_map": {"a.weight": "model.safetensors"}
            }"#,
        )
        .unwrap();

        let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
        assert!(
            !manifest
                .checkpoint
                .metadata
                .contains_key(INDEX_UNREFERENCED_SHARDS_KEY)
        );
    }

    #[test]
    fn index_shard_paths_are_normalized_before_deduplication() {
        let dir = temp_dir("normalized-index-paths");
        write_safetensors(
            &dir.join("model.safetensors"),
            r#"{
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]}
            }"#,
            4,
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{
                "weight_map": {
                    "a.weight": "model.safetensors",
                    "b.weight": "./model.safetensors"
                }
            }"#,
        )
        .unwrap();

        let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
        assert_eq!(manifest.checkpoint.shard_count, 1);
        assert_eq!(manifest.checkpoint.tensor_count, 2);
        assert_eq!(manifest.tensors[0].source_shard, "model.safetensors");
        assert_eq!(manifest.tensors[1].source_shard, "model.safetensors");
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

    #[cfg(unix)]
    #[test]
    fn rejects_directory_index_that_escapes_via_symlink() {
        let dir = temp_dir("index-symlink");
        let outside = temp_dir("outside-index");
        fs::write(
            outside.join("external.safetensors.index.json"),
            r#"{"weight_map": {}}"#,
        )
        .unwrap();
        std::os::unix::fs::symlink(
            outside.join("external.safetensors.index.json"),
            dir.join("model.safetensors.index.json"),
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
        write_safetensors(
            &dir.join("unused.safetensors"),
            r#"{"unused.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
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
        assert_eq!(
            manifest
                .checkpoint
                .metadata
                .get(INDEX_UNREFERENCED_SHARDS_KEY)
                .map(String::as_str),
            Some(r#"["unused.safetensors"]"#)
        );

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
    fn rejects_unknown_safetensors_dtype() {
        let dir = temp_dir("unknown-dtype");
        let path = dir.join("bad.safetensors");
        write_safetensors(
            &path,
            r#"{"bad.weight": {"dtype": "F128", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );

        let err = inspect_safetensors_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported Safetensors dtype"));
    }

    #[test]
    fn rejects_duplicate_header_keys() {
        let dir = temp_dir("duplicate-keys");
        let path = dir.join("bad.safetensors");
        write_safetensors(
            &path,
            r#"{
                "dup.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "dup.weight": {"dtype": "F16", "shape": [1], "data_offsets": [2, 4]}
            }"#,
            4,
        );

        let err = inspect_safetensors_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("duplicate JSON key"));
    }

    #[test]
    fn rejects_non_string_metadata_values() {
        let dir = temp_dir("metadata-values");
        let path = dir.join("bad.safetensors");
        write_safetensors(
            &path,
            r#"{
                "__metadata__": {"format": 1},
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            }"#,
            2,
        );

        let err = inspect_safetensors_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("must be a string"));
    }

    #[test]
    fn rejects_overlapping_tensor_ranges() {
        let dir = temp_dir("overlap");
        let path = dir.join("bad.safetensors");
        write_safetensors(
            &path,
            r#"{
                "a.weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
                "b.weight": {"dtype": "F16", "shape": [4], "data_offsets": [4, 12]}
            }"#,
            12,
        );

        let err = inspect_safetensors_checkpoint(&path).unwrap_err();
        assert!(err.to_string().contains("data range overlaps"));
    }

    #[test]
    fn rejects_tensor_data_gaps_and_trailing_bytes() {
        let dir = temp_dir("data-gaps");
        let gap_path = dir.join("gap.safetensors");
        write_safetensors(
            &gap_path,
            r#"{
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
                "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [4, 6]}
            }"#,
            6,
        );
        let err = inspect_safetensors_checkpoint(&gap_path).unwrap_err();
        assert!(err.to_string().contains("leaves a gap"));

        let trailing_path = dir.join("trailing.safetensors");
        write_safetensors(
            &trailing_path,
            r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            4,
        );
        let err = inspect_safetensors_checkpoint(&trailing_path).unwrap_err();
        assert!(err.to_string().contains("data section is 4 bytes"));
    }

    #[test]
    fn conflicting_shard_metadata_is_only_namespaced() {
        let dir = temp_dir("metadata-conflict");
        write_safetensors(
            &dir.join("a.safetensors"),
            r#"{
                "__metadata__": {"format": "pt"},
                "a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            }"#,
            2,
        );
        write_safetensors(
            &dir.join("b.safetensors"),
            r#"{
                "__metadata__": {"format": "np"},
                "b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}
            }"#,
            2,
        );

        let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
        assert!(!manifest.checkpoint.metadata.contains_key("format"));
        assert_eq!(
            manifest
                .checkpoint
                .metadata
                .get(&shard_metadata_namespaced_key("a.safetensors", "format"))
                .map(String::as_str),
            Some("pt")
        );
        assert_eq!(
            manifest
                .checkpoint
                .metadata
                .get(&shard_metadata_namespaced_key("b.safetensors", "format"))
                .map(String::as_str),
            Some("np")
        );
    }

    #[test]
    fn colon_in_metadata_logical_key_conflict_does_not_restore_base_key() {
        let mut metadata = BTreeMap::new();
        let mut first = BTreeMap::new();
        first.insert("my:key".to_string(), "a".to_string());
        merge_shard_metadata(&mut metadata, "x.safetensors", first);

        let mut second = BTreeMap::new();
        second.insert("my:key".to_string(), "b".to_string());
        merge_shard_metadata(&mut metadata, "y.safetensors", second);

        assert!(!metadata.contains_key("my:key"));

        let mut third = BTreeMap::new();
        third.insert("my:key".to_string(), "c".to_string());
        merge_shard_metadata(&mut metadata, "z.safetensors", third);

        assert!(!metadata.contains_key("my:key"));
        assert_eq!(
            metadata
                .get(&shard_metadata_namespaced_key("z.safetensors", "my:key"))
                .map(String::as_str),
            Some("c")
        );
    }

    #[test]
    fn rejects_manifest_output_that_overwrites_unreferenced_index_shard_with_comma_in_name() {
        let dir = temp_dir("output-unreferenced-comma");
        let main_shard = dir.join("model-00001-of-00001.safetensors");
        write_safetensors(
            &main_shard,
            r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );
        let unreferenced = dir.join("a,b.safetensors");
        write_safetensors(
            &unreferenced,
            r#"{"b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let err = write_safetensors_manifest(&dir, &unreferenced).unwrap_err();
        assert!(err.to_string().contains("must not overwrite"));
    }

    #[test]
    fn shard_metadata_key_detection_uses_exact_key_match() {
        let mut metadata = BTreeMap::new();
        let mut first = BTreeMap::new();
        first.insert("xfoo".to_string(), "1".to_string());
        merge_shard_metadata(&mut metadata, "a.safetensors", first);

        let mut second = BTreeMap::new();
        second.insert("foo".to_string(), "2".to_string());
        merge_shard_metadata(&mut metadata, "b.safetensors", second);

        assert_eq!(metadata.get("foo").map(String::as_str), Some("2"));
    }

    #[test]
    fn directory_scan_ignores_non_file_safetensors_entries() {
        let dir = temp_dir("non-file-entry");
        fs::create_dir(dir.join("scratch.safetensors")).unwrap();
        write_safetensors(
            &dir.join("model.safetensors"),
            r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );

        let manifest = inspect_safetensors_checkpoint(&dir).unwrap();
        assert_eq!(manifest.checkpoint.shard_count, 1);
        assert_eq!(manifest.checkpoint.tensor_count, 1);
        assert_eq!(manifest.tensors[0].source_shard, "model.safetensors");
    }

    #[test]
    fn bare_paths_resolve_parent_as_current_directory() {
        assert_eq!(
            parent_or_current(Path::new("model.safetensors")),
            Path::new(".")
        );
        assert!(canonical_existing_or_parent(Path::new("manifest.json")).is_ok());
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
    fn rejects_manifest_output_that_overwrites_unreferenced_index_shard() {
        let dir = temp_dir("output-unreferenced");
        let main_shard = dir.join("model-00001-of-00001.safetensors");
        write_safetensors(
            &main_shard,
            r#"{"a.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );
        let unreferenced = dir.join("unused.safetensors");
        write_safetensors(
            &unreferenced,
            r#"{"b.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}"#,
            2,
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a.weight": "model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();

        let err = write_safetensors_manifest(&dir, &unreferenced).unwrap_err();
        assert!(err.to_string().contains("must not overwrite"));
    }

    #[test]
    fn accepts_current_additional_safetensors_dtypes() {
        let dir = temp_dir("extra-dtypes");
        let path = dir.join("model.safetensors");
        write_safetensors(
            &path,
            r#"{
                "a.weight": {"dtype": "F8_E8M0", "shape": [4], "data_offsets": [0, 4]},
                "b.weight": {"dtype": "C64", "shape": [2], "data_offsets": [4, 20]}
            }"#,
            20,
        );

        let manifest = inspect_safetensors_checkpoint(&path).unwrap();
        assert_eq!(manifest.checkpoint.tensor_count, 2);
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
