mod support;

use corinth_canal::{gpu::GpuAccelerator, model::Model};
use std::io::Error;
use std::time::Instant;
use support::{
    DEFAULT_MATH_PROMPT_TOKEN_IDS, default_spiking_model_config, mean_pool_prompt_embeddings,
    required_gguf_checkpoint_path,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = required_gguf_checkpoint_path()?;

    let mut accelerator = GpuAccelerator::new();
    let mut model = Model::new(default_spiking_model_config(model_path.clone(), 1))?;

    if !model.router_loaded() {
        return Err(Error::other("OlmoeRouter model did not load from GGUF_CHECKPOINT_PATH").into());
    }
    if !accelerator.is_ready() {
        return Err(Error::other("GpuAccelerator is not ready").into());
    }

    // 1. Extract embeddings for token IDs: [1045, 2099, 450, 8000, 12]
    // These represent the prompt: "Let's teach this MoE model the language of SNN"
    let pooled = mean_pool_prompt_embeddings(&mut model, &DEFAULT_MATH_PROMPT_TOKEN_IDS)?;

    let target_neurons = model.projector_mut().input_neurons();
    if pooled.len() != target_neurons {
        return Err(Error::other(format!(
            "Dimension mismatch: pooled len {} != target_neurons {}",
            pooled.len(),
            target_neurons
        ))
        .into());
    }

    // 3. Prepare GPU temporal state
    model.prepare_gpu_temporal(&mut accelerator)?;

    // 4. Run the 10,000-tick loop using the single pooled context vector continuously
    for tick in 0..10_000usize {
        let started = Instant::now();
        let best_walker = model.tick_gpu_temporal(&mut accelerator, &pooled)?;
        let elapsed_us = started.elapsed().as_micros();
        println!(
            "tick={} best_walker={} elapsed_us={}",
            tick + 1,
            best_walker,
            elapsed_us
        );
    }

    // Maintain strict drop order
    drop(model);
    drop(accelerator);

    Ok(())
}
