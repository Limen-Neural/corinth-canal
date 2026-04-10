//! CSV replay example: ingest canonical telemetry CSV into HybridModel.
//!
//! Canonical CSV format: timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w

use corinth_canal::{
    EMBEDDING_DIM, HybridConfig, HybridError, HybridModel, OlmoeExecutionMode, ProjectionMode,
    TelemetrySnapshot,
};

const EXPECTED_HEADER: &str =
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";

fn parse_u64(v: &str) -> Option<u64> {
    v.parse::<u64>().ok()
}

fn parse_f32(v: &str) -> Option<f32> {
    let n = v.parse::<f32>().ok()?;
    if n.is_finite() {
        Some(n)
    } else {
        None
    }
}

fn main() -> corinth_canal::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example csv_replay <telemetry.csv>");
        eprintln!("  CSV format: {}", EXPECTED_HEADER);
        std::process::exit(1);
    }

    let csv_path = &args[1];
    let model_path = std::env::var("OLMOE_PATH").unwrap_or_default();

    let cfg = HybridConfig {
        olmoe_model_path: model_path,
        snn_steps: 20,
        num_experts: 8,
        top_k_experts: 1,
        olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
        projection_mode: ProjectionMode::SpikingTernary,
    };

    let mut model = HybridModel::new(cfg)?;
    println!(
        "olmoe_loaded={} olmoe_mode={:?}",
        model.olmoe_loaded(),
        model.config().olmoe_execution_mode
    );

    let csv_content = std::fs::read_to_string(csv_path)?;
    let mut lines = csv_content.lines();

    let header = lines
        .next()
        .ok_or_else(|| HybridError::InvalidConfig("empty CSV file".to_owned()))?
        .trim();

    if header != EXPECTED_HEADER {
        return Err(HybridError::InvalidConfig(format!(
            "invalid CSV header: expected '{EXPECTED_HEADER}', got '{header}'"
        )));
    }

    let mut total_loss = 0.0_f32;
    let mut rows_processed = 0_usize;
    let mut rows_skipped = 0_usize;

    for (idx, raw_line) in lines.enumerate() {
        let line_number = idx + 2;
        let line = raw_line.trim();

        if line.is_empty() {
            rows_skipped += 1;
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 5 {
            rows_skipped += 1;
            eprintln!(
                "Skipping malformed row {}: expected 5 columns, got {}",
                line_number,
                fields.len()
            );
            continue;
        }

        let parsed = (
            parse_u64(fields[0]),
            parse_f32(fields[1]),
            parse_f32(fields[2]),
            parse_f32(fields[3]),
            parse_f32(fields[4]),
        );

        let (Some(timestamp_ms), Some(gpu_temp_c), Some(gpu_power_w), Some(cpu_tctl_c), Some(cpu_package_power_w)) =
            parsed
        else {
            rows_skipped += 1;
            eprintln!("Skipping malformed row {}: parse/finite check failed", line_number);
            continue;
        };

        let snap = TelemetrySnapshot {
            timestamp_ms,
            gpu_temp_c,
            gpu_power_w,
            cpu_tctl_c,
            cpu_package_power_w,
        };

        let output = model.forward(&snap)?;

        let mean_embed = output.embedding.iter().sum::<f32>() / EMBEDDING_DIM as f32;
        let target = vec![mean_embed * 0.9; EMBEDDING_DIM];
        let loss = output
            .embedding
            .iter()
            .zip(target.iter())
            .map(|(hidden, expected)| (hidden - expected).powi(2))
            .sum::<f32>()
            / EMBEDDING_DIM as f32;

        total_loss += loss;
        rows_processed += 1;

        if rows_processed % 100 == 0 || rows_processed <= 5 {
            println!(
                "step={:>4} gpu_temp={:5.1}C gpu_power={:6.1}W cpu_temp={:5.1}C loss={:.6}",
                rows_processed, gpu_temp_c, gpu_power_w, cpu_tctl_c, loss
            );
        }
    }

    let avg_loss = if rows_processed > 0 {
        total_loss / rows_processed as f32
    } else {
        0.0
    };

    println!("\n=== Replay Summary ===");
    println!("rows_processed={}", rows_processed);
    println!("rows_skipped={}", rows_skipped);
    println!("avg_loss={:.6}", avg_loss);
    println!("global_step={}", model.global_step());
    println!("olmoe_loaded={}", model.olmoe_loaded());

    Ok(())
}
