//! CSV replay example: ingest canonical telemetry CSV into HybridModel.
//!
//! Canonical CSV format: timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w

use corinth_canal::{
    EMBEDDING_DIM, HybridConfig, HybridModel, OlmoeExecutionMode, ProjectionMode,
    TelemetrySnapshot,
};

fn main() -> corinth_canal::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example csv_replay <telemetry.csv>");
        eprintln!("  CSV format: timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w");
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

    // Parse CSV
    let csv_content = std::fs::read_to_string(csv_path)?;
    let mut lines = csv_content.lines();

    // Skip header
    let header = lines.next().ok_or("Empty CSV file")?;
    if !header.contains("timestamp_ms") {
        eprintln!("Warning: CSV header may be missing expected columns");
    }

    let mut total_loss = 0.0_f32;
    let mut row_count = 0_usize;

    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            continue; // Skip malformed lines
        }

        let timestamp_ms: u64 = fields[0].parse().unwrap_or(0);
        let gpu_temp_c: f32 = fields[1].parse().unwrap_or(0.0);
        let gpu_power_w: f32 = fields[2].parse().unwrap_or(0.0);
        let cpu_tctl_c: f32 = fields[3].parse().unwrap_or(0.0);
        let cpu_package_power_w: f32 = fields[4].parse().unwrap_or(0.0);

        let snap = TelemetrySnapshot {
            timestamp_ms,
            gpu_temp_c,
            gpu_power_w,
            cpu_tctl_c,
            cpu_package_power_w,
        };

        let output = model.forward(&snap)?;

        // Calculate simple loss against mean embedding
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
        row_count += 1;

        // Per-step diagnostics
        if row_count % 100 == 0 || row_count <= 5 {
            println!(
                "step={:>4} gpu_temp={:5.1}C gpu_power={:6.1}W cpu_temp={:5.1}C loss={:.6}",
                row_count, gpu_temp_c, gpu_power_w, cpu_tctl_c, loss
            );
        }
    }

    // Final summary
    let avg_loss = if row_count > 0 { total_loss / row_count as f32 } else { 0.0 };

    println!("\n=== Replay Summary ===");
    println!("Rows processed: {}", row_count);
    println!("Avg loss: {:.6}", avg_loss);
    println!("Global step: {}", model.global_step());
    println!("OLMoE loaded: {}", model.olmoe_loaded());

    Ok(())
}
