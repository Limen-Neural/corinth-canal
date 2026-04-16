# SAAQ Latent Space Calibration

Research log documenting the stabilization of hybrid LLM-SNN inference through neuromorphic normalization techniques.

---

## Phase 1: Synthetic Baseline

![Smoke Test](latent_space_exploration_smoketest.png)

I verified the bare-metal GPU temporal loop using a synthetic sine wave. The Spiking Neural Network (SNN) successfully demonstrated basic biological fatigue and routing without crashing the CUDA context.

---

## Phase 2: The F16 Magnitude Collapse

![Routing Collapse](latent_space_exploration_first_real_attempt.png)

I fed real LLM embeddings from the OLMoE GGUF file. The weights were safely cast from F16 to F32, but the raw, unbounded electrical pressure of the semantic vector overpowered the network's fatigue mechanics. This resulted in a 'Winner-Take-All' routing collapse where a single neuron took the entire load for 10,000 ticks.

---

## Phase 3: L2 Normalization and Attractor Discovery

![Attractor Discovery](latent_space_exploration_2nd_attempt_successful.png)

I applied L2 Normalization to the pooled embedding vector, bounding the semantic pressure to a unit sphere. This successfully shattered the routing collapse, forcing the SNN to organically balance the load. The resulting plot shows clear, discrete 'Attractor Bands', proving the neuromorphic engine can physically route and hold an LLM concept in stable working memory.

---

## Summary

| Phase | Input | Result | Key Learning |
|-------|-------|--------|--------------|
| 1 | Synthetic sine wave | Stable routing loop | CUDA/GPU baseline functional |
| 2 | Raw LLM embeddings (F16→F32) | Routing collapse | Unbounded magnitude destroys fatigue balance |
| 3 | L2-normalized embeddings | Attractor bands emerge | Unit sphere projection enables stable SNN working memory |

**Conclusion:** L2 normalization is a mandatory preprocessing step for LLM-SNN quantization architectures to prevent magnitude-induced routing collapse and enable stable attractor-based computation.
