mod support;

use corinth_canal::{gpu::GpuAccelerator, model::Model};
use std::io::Error;
use std::time::Instant;
use support::{default_spiking_model_config, required_gguf_checkpoint_path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = required_gguf_checkpoint_path()?;

    let mut accelerator = GpuAccelerator::new();
    let mut model = Model::new(default_spiking_model_config(model_path.clone(), 1))?;

    let target_neurons = model.projector_mut().input_neurons();
    println!(
        "startup model_path={} router_loaded={} gpu_ready={} target_neurons={}",
        model_path,
        model.router_loaded(),
        accelerator.is_ready(),
        target_neurons,
    );

    if !model.router_loaded() {
        return Err(Error::other("OlmoeRouter model did not load from GGUF_CHECKPOINT_PATH").into());
    }
    if !accelerator.is_ready() {
        return Err(Error::other("GpuAccelerator is not ready").into());
    }

    model.prepare_gpu_temporal(&mut accelerator)?;
    println!("prepared gguf-backed temporal path; beginning 10,000 direct GPU ticks");

    for tick in 0..10_000usize {
        let phase = tick as f32 * 0.31;
        let input_spikes: Vec<f32> = (0..target_neurons)
            .map(|i| {
                let wave = (i as f32 * 0.017 + phase).sin();
                0.1 * (wave + 1.0) * 0.5
            })
            .collect();

        let started = Instant::now();
        let best_walker = model.tick_gpu_temporal(&mut accelerator, &input_spikes)?;
        let elapsed_us = started.elapsed().as_micros();
        println!(
            "tick={} best_walker={} elapsed_us={}",
            tick + 1,
            best_walker,
            elapsed_us
        );
    }

    println!("completed 10,000 GPU ticks; dropping model before accelerator");
    drop(model);
    drop(accelerator);
    println!("gpu smoke test finished cleanly");

    Ok(())
}
