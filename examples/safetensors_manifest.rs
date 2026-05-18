//! Generate a deterministic JSON manifest from Safetensors checkpoint headers.
//!
//! Usage:
//!
//! ```text
//! cargo run --example safetensors_manifest --no-default-features -- <checkpoint-or-dir> <output.json>
//! ```

use corinth_canal::moe::safetensors::write_safetensors_manifest;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let checkpoint_path = args.next().map(PathBuf::from).ok_or_else(|| {
        "usage: safetensors_manifest <checkpoint.safetensors|index.json|directory> <output.json>"
            .to_string()
    })?;
    let output_path = args.next().map(PathBuf::from).ok_or_else(|| {
        "usage: safetensors_manifest <checkpoint.safetensors|index.json|directory> <output.json>"
            .to_string()
    })?;
    if args.next().is_some() {
        return Err(
            "usage: safetensors_manifest <checkpoint.safetensors|index.json|directory> <output.json>"
                .into(),
        );
    }

    if let Some(parent) = output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let manifest = write_safetensors_manifest(&checkpoint_path, &output_path)?;
    println!(
        "wrote {} tensors from {} shard(s) to {}",
        manifest.checkpoint.tensor_count,
        manifest.checkpoint.shard_count,
        output_path.display()
    );
    println!(
        "router_candidates={} expert_candidates={}",
        manifest.candidates.router_tensors.len(),
        manifest.candidates.expert_tensors.len()
    );

    Ok(())
}
