mod support;

use corinth_canal::{
    FunnelActivity, HeartbeatInjector, SaaqUpdateRule, SnnLatentCalibrator, SnnLatentCsvExporter,
    gpu::GpuAccelerator, model::Model,
};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Error, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use support::{
    ValidationModelSpec, default_spiking_model_config, discover_validation_models, heartbeat_gain,
    heartbeat_modes_for_matrix, model_family_override_from_env,
    prompt_embedding_for_validation, prompt_profile_slug, prompt_text_for_profile,
    saaq_update_rule_from_env, synthetic_base_snapshot, ticks_from_env,
};

#[derive(Debug, Serialize)]
struct ValidationManifest {
    model_slug: String,
    model_family: String,
    architecture: String,
    checkpoint_path: String,
    routing_tensor_name: String,
    synapse_source: String,
    checkpoint_format: &'static str,
    prompt_embedding_source: String,
    prompt_profile: String,
    prompt_text: String,
    ticks: usize,
    saaq_rule: &'static str,
    validation_status: &'static str,
    error: Option<String>,
    heartbeat_enabled: bool,
    heartbeat_amplitude: f32,
    heartbeat_period_ticks: usize,
    heartbeat_duty_cycle: f32,
    heartbeat_phase_offset_ticks: usize,
    generated_files: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt_profile = prompt_profile_slug();
    let prompt_text = prompt_text_for_profile(&prompt_profile);
    let ticks = ticks_from_env(512);
    let models = discover_validation_models();
    if models.is_empty() {
        return Err(Error::other(
            "No GGUF validation models discovered. Set GGUF_CHECKPOINT_PATH or place models under ~/Downloads/SNN_Quantization.",
        )
        .into());
    }

    for spec in models {
        for heartbeat_enabled in heartbeat_modes_for_matrix() {
            run_validation(&spec, &prompt_profile, prompt_text, ticks, heartbeat_enabled)?;
        }
    }

    Ok(())
}

fn run_validation(
    spec: &ValidationModelSpec,
    prompt_profile: &str,
    prompt_text: &str,
    ticks: usize,
    heartbeat_enabled: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = default_spiking_model_config(spec.path.clone(), 1);
    config.model_family = model_family_override_from_env().or(spec.family);
    config.heartbeat.enabled = heartbeat_enabled;
    let saaq_rule = saaq_update_rule_from_env();

    let mut accelerator = GpuAccelerator::new();
    let mut model = Model::new(config.clone())?;

    if !model.router_loaded() {
        return Err(Error::other(format!(
            "router did not load for checkpoint '{}'",
            spec.path
        ))
        .into());
    }

    let run_dir = build_run_dir(spec, prompt_profile, heartbeat_enabled)?;
    fs::create_dir_all(&run_dir)?;
    let tick_path = run_dir.join("tick_telemetry.txt");
    let latent_path = run_dir.join("latent_telemetry.csv");
    let manifest_path = run_dir.join("run_manifest.json");
    let generated_files = vec![
        tick_path.file_name().unwrap().to_string_lossy().into_owned(),
        latent_path.file_name().unwrap().to_string_lossy().into_owned(),
        manifest_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
    ];
    let target_neurons = model.projector_mut().input_neurons();
    let (prompt_embedding, prompt_embedding_source) =
        match prompt_embedding_for_validation(&spec.path, prompt_text, target_neurons) {
            Ok(result) => result,
            Err(error) => {
                write_manifest(
                    &manifest_path,
                    build_manifest(
                        spec,
                        prompt_profile,
                        prompt_text,
                        ticks,
                        &config,
                        &model,
                        heartbeat_enabled,
                        "unavailable",
                        saaq_rule,
                        "prompt_embedding_failed",
                        Some(error.to_string()),
                        generated_files.clone(),
                    ),
                )?;
                return Err(error);
            }
        };

    write_manifest(
        &manifest_path,
        build_manifest(
            spec,
            prompt_profile,
            prompt_text,
            ticks,
            &config,
            &model,
            heartbeat_enabled,
            &prompt_embedding_source,
            saaq_rule,
            "preflight",
            None,
            generated_files.clone(),
        ),
    )?;

    if let Err(error) = model.prepare_gpu_temporal(&mut accelerator) {
        write_manifest(
            &manifest_path,
            build_manifest(
                spec,
                prompt_profile,
                prompt_text,
                ticks,
                &config,
                &model,
                heartbeat_enabled,
                &prompt_embedding_source,
                saaq_rule,
                "gpu_setup_failed",
                Some(error.to_string()),
                generated_files,
            ),
        )?;
        return Err(Box::new(error));
    }

    let mut tick_writer = BufWriter::new(File::create(&tick_path)?);
    let mut latent_exporter = SnnLatentCsvExporter::create(&latent_path)?;
    let mut calibrator = SnnLatentCalibrator::with_update_rule(saaq_rule);
    let heartbeat = HeartbeatInjector::new(config.heartbeat.clone());

    println!(
        "validation_start model_slug={} family={:?} architecture={} heartbeat_enabled={} ticks={} routing_tensor={} synapse_source={}",
        spec.slug,
        model.router_family(),
        model.router_architecture(),
        heartbeat_enabled,
        ticks,
        model.routing_tensor_name(),
        model.synapse_source(),
    );

    let run_result = (|| -> Result<(), Box<dyn std::error::Error>> {
        for tick in 0..ticks {
            let mut snap = synthetic_base_snapshot(tick);
            snap.timestamp_ms = (tick as u64) + 1;
            let snap = heartbeat.apply(&snap, tick);
            let gain = heartbeat_gain(snap.heartbeat_signal);
            let input_spikes: Vec<f32> =
                prompt_embedding.iter().map(|value| value * gain).collect();

            let started = Instant::now();
            let best_walker = model.tick_gpu_temporal(&mut accelerator, &input_spikes)?;
            let elapsed_us = started.elapsed().as_micros();

            let spikes = accelerator.temporal_spikes_to_vec(target_neurons)?;
            let active_neurons: Vec<usize> = spikes
                .iter()
                .enumerate()
                .filter(|(_, value)| **value != 0)
                .map(|(idx, _)| idx)
                .collect();
            let potentials = accelerator
                .temporal_membrane_to_vec(target_neurons)?
                .into_iter()
                .map(|value| value.clamp(0.0, 1.0))
                .collect::<Vec<f32>>();
            let activity = FunnelActivity {
                ternary_events: [0; 4],
                input_spike_train: vec![active_neurons.clone()],
                spike_train: vec![active_neurons],
                potentials: potentials.clone(),
                iz_potentials: vec![0.0; 5],
            };
            let output = model.forward_activity(
                &activity.spike_train,
                &activity.potentials,
                &activity.iz_potentials,
            )?;
            let latent = calibrator.observe(&snap, &activity, &output)?;
            latent_exporter.write_row(&latent)?;

            writeln!(
                tick_writer,
                "tick={} best_walker={} elapsed_us={} heartbeat_signal={:.6} gpu_temp_c={:.3} gpu_power_w={:.3} cpu_tctl_c={:.3} cpu_package_power_w={:.3}",
                tick + 1,
                best_walker,
                elapsed_us,
                snap.heartbeat_signal,
                snap.gpu_temp_c,
                snap.gpu_power_w,
                snap.cpu_tctl_c,
                snap.cpu_package_power_w,
            )?;
        }

        latent_exporter.flush()?;
        tick_writer.flush()?;
        Ok(())
    })();

    if let Err(error) = run_result {
        let _ = latent_exporter.flush();
        let _ = tick_writer.flush();
        write_manifest(
            &manifest_path,
            build_manifest(
                spec,
                prompt_profile,
                prompt_text,
                ticks,
                &config,
                &model,
                heartbeat_enabled,
                &prompt_embedding_source,
                saaq_rule,
                "tick_failed",
                Some(error.to_string()),
                generated_files.clone(),
            ),
        )?;
        return Err(error);
    }

    let manifest = ValidationManifest {
        model_slug: spec.slug.clone(),
        model_family: format!("{:?}", model.router_family()),
        architecture: model.router_architecture().to_owned(),
        checkpoint_path: spec.path.clone(),
        routing_tensor_name: model.routing_tensor_name().to_owned(),
        synapse_source: model.synapse_source().to_owned(),
        checkpoint_format: "gguf",
        prompt_embedding_source,
        prompt_profile: prompt_profile.to_owned(),
        prompt_text: prompt_text.to_owned(),
        ticks,
        saaq_rule: saaq_rule_label(saaq_rule),
        validation_status: "completed",
        error: None,
        heartbeat_enabled,
        heartbeat_amplitude: config.heartbeat.amplitude,
        heartbeat_period_ticks: config.heartbeat.period_ticks,
        heartbeat_duty_cycle: config.heartbeat.duty_cycle,
        heartbeat_phase_offset_ticks: config.heartbeat.phase_offset_ticks,
        generated_files,
    };
    write_manifest(&manifest_path, manifest)?;

    println!(
        "validation_complete model_slug={} heartbeat_enabled={} run_dir={}",
        spec.slug,
        heartbeat_enabled,
        run_dir.display()
    );

    drop(model);
    drop(accelerator);

    Ok(())
}

fn build_manifest(
    spec: &ValidationModelSpec,
    prompt_profile: &str,
    prompt_text: &str,
    ticks: usize,
    config: &corinth_canal::model::ModelConfig,
    model: &Model,
    heartbeat_enabled: bool,
    prompt_embedding_source: &str,
    saaq_rule: SaaqUpdateRule,
    validation_status: &'static str,
    error: Option<String>,
    generated_files: Vec<String>,
) -> ValidationManifest {
    ValidationManifest {
        model_slug: spec.slug.clone(),
        model_family: format!("{:?}", model.router_family()),
        architecture: model.router_architecture().to_owned(),
        checkpoint_path: spec.path.clone(),
        routing_tensor_name: model.routing_tensor_name().to_owned(),
        synapse_source: model.synapse_source().to_owned(),
        checkpoint_format: "gguf",
        prompt_embedding_source: prompt_embedding_source.to_owned(),
        prompt_profile: prompt_profile.to_owned(),
        prompt_text: prompt_text.to_owned(),
        ticks,
        saaq_rule: saaq_rule_label(saaq_rule),
        validation_status,
        error,
        heartbeat_enabled,
        heartbeat_amplitude: config.heartbeat.amplitude,
        heartbeat_period_ticks: config.heartbeat.period_ticks,
        heartbeat_duty_cycle: config.heartbeat.duty_cycle,
        heartbeat_phase_offset_ticks: config.heartbeat.phase_offset_ticks,
        generated_files,
    }
}

fn write_manifest(
    manifest_path: &PathBuf,
    manifest: ValidationManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(manifest_path, serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

fn saaq_rule_label(rule: SaaqUpdateRule) -> &'static str {
    match rule {
        SaaqUpdateRule::LegacyV1_0 => "LegacyV1_0",
        SaaqUpdateRule::SaaqV1_5SqrtRate => "SaaqV1_5SqrtRate",
    }
}

fn build_run_dir(
    spec: &ValidationModelSpec,
    prompt_profile: &str,
    heartbeat_enabled: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let root = std::env::var("VALIDATION_OUTPUT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("outputs").join("validation"));
    let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let heartbeat_slug = if heartbeat_enabled {
        "heartbeat_on"
    } else {
        "heartbeat_off"
    };
    Ok(root.join(format!(
        "{epoch}_{}_{}_{}",
        spec.slug, prompt_profile, heartbeat_slug
    )))
}
