use corinth_canal::{
    FUNNEL_HIDDEN_NEURONS, HybridConfig, HybridError, HybridModel, OlmoeExecutionMode,
    ProjectionMode, SnnLatentCalibrator, SnnLatentCsvExporter, TelemetryFunnel,
    TelemetrySnapshot,
};

const EXPECTED_HEADER: &str =
    "timestamp_ms,gpu_temp_c,gpu_power_w,cpu_tctl_c,cpu_package_power_w";
const TELEMETRY_THRESHOLDS: [f32; 4] = [1.0, 5.0, 1.0, 5.0];

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
    if args.len() < 2 || args.len() > 3 {
        eprintln!(
            "Usage: cargo run --example saaq_latent_calibration <telemetry.csv> [snn_latent_telemetry.csv]"
        );
        eprintln!("  CSV format: {}", EXPECTED_HEADER);
        std::process::exit(1);
    }

    let csv_path = &args[1];
    let output_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "snn_latent_telemetry.csv".to_owned());
    let model_path = std::env::var("OLMOE_PATH").unwrap_or_default();

    let cfg = HybridConfig {
        olmoe_model_path: model_path,
        snn_steps: 20,
        num_experts: 8,
        top_k_experts: 1,
        olmoe_execution_mode: OlmoeExecutionMode::SpikingSim,
        projection_mode: ProjectionMode::SpikingTernary,
    };

    let mut model = HybridModel::new_with_projector_neurons(cfg.clone(), FUNNEL_HIDDEN_NEURONS)?;
    let mut funnel = TelemetryFunnel::new(TELEMETRY_THRESHOLDS, cfg.snn_steps);
    let mut calibrator = SnnLatentCalibrator::new();
    let mut exporter = SnnLatentCsvExporter::create(&output_path)?;

    println!(
        "olmoe_loaded={} olmoe_mode={:?} funnel_hidden_neurons={} latent_output={}",
        model.olmoe_loaded(),
        model.config().olmoe_execution_mode,
        FUNNEL_HIDDEN_NEURONS,
        output_path,
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

        let (
            Some(timestamp_ms),
            Some(gpu_temp_c),
            Some(gpu_power_w),
            Some(cpu_tctl_c),
            Some(cpu_package_power_w),
        ) = parsed
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

        let activity = funnel.encode_snapshot(&snap);
        let output = model.forward_activity(
            &activity.spike_train,
            &activity.potentials,
            &activity.iz_potentials,
        )?;
        let latent = calibrator.observe(&snap, &activity, &output)?;
        exporter.write_row(&latent)?;
        rows_processed += 1;

        if rows_processed % 100 == 0 || rows_processed <= 5 {
            println!(
                "step={:>4} avg_pop_firing_rate_hz={:.6} membrane_dv_dt={:.6} routing_entropy={:.6} saaq_delta_q_prev={:.6} saaq_delta_q_target={:.6}",
                rows_processed,
                latent.avg_pop_firing_rate_hz,
                latent.membrane_dv_dt,
                latent.routing_entropy,
                latent.saaq_delta_q_prev,
                latent.saaq_delta_q_target,
            );
        }
    }

    exporter.flush()?;

    println!("\n=== Latent Calibration Summary ===");
    println!("rows_processed={}", rows_processed);
    println!("rows_skipped={}", rows_skipped);
    println!("global_step={}", model.global_step());
    println!("olmoe_loaded={}", model.olmoe_loaded());
    println!("latent_csv={}", output_path);

    Ok(())
}
