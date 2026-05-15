# CSV/off-only SAAQ summary

This intentionally ignores synthetic fallback and heartbeat-on runs.


| model | repeat | status | rows | entropy mean | entropy min | entropy max | entropy range | v1.5 final | v1.5 range | legacy final | legacy range |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4_26b_a4b_iq4_nl | 0 | completed | 2000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0217 |
| gemma4_26b_a4b_iq4_nl | 1 | completed | 2000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0217 |
| llama_3_2_dark_champion_q5_k_m | 0 | completed | 2000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0217 |
| llama_3_2_dark_champion_q5_k_m | 1 | completed | 2000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0217 |
| olmoe_baseline | 0 | completed | 2000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0310 |
| olmoe_baseline | 1 | completed | 2000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0417 | 0.0310 |

## Interpretation checklist


- If entropy range is near zero, routing behavior may be flat.
- If SAAQ ranges are near zero, SAAQ response may be flat.
- If repeats match closely, the run is deterministic enough for comparison.
- Synthetic and heartbeat runs are intentionally excluded from this first pass.
