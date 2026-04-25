# Known-Good Runs

Append-only log. One entry per run ID that has been hand-reviewed and
blessed as reference material for future promotion. Latest entries at the
top.

Format:

```
## <run_id>
- checkpoint: <model_slug> (<family>)
- telemetry:  <source_label>
- heartbeat:  on|off
- saaq_rule:  saaq_v1_5 | legacy
- conclusion: <one line>
- artifacts:  <path under VALIDATION_OUTPUT_ROOT, or "artifacts/<run_id>/">
```

---

_No blessed runs yet. Bootstrap this file once Stage E is verified and the
first artifacts/ tree is reviewed._

## SAAQ 1.5 OLMoE RE4 Control — 2026-04-23

- Model: `olmoe_baseline`
- Family: `Olmoe`
- Rule: `SaaqV1_5SqrtRate`
- Telemetry: `csv_re4_path_tracing_telemetry`
- Heartbeat: `off`
- Repeat count: `2`
- Determinism: `matched`
- Rows: `2000`
- Run 0: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_off/20260423T195615_math_logic_r0_baseline_csv_off`
- Run 1: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_off/20260423T195637_math_logic_r1_baseline_csv_off`

Conclusion: heartbeat-off SAAQ 1.5 control baseline completed successfully on OLMoE with matched repeat determinism.

## SAAQ 1.5 OLMoE RE4 Baseline — 2026-04-23

- Model: `olmoe_baseline`
- Family: `Olmoe`
- Rule: `SaaqV1_5SqrtRate`
- Telemetry: `csv_re4_path_tracing_telemetry`
- Heartbeat: `on`
- Repeat count: `2`
- Determinism: `matched`
- Rows: `2000`
- Run 0: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_on/20260423T195816_math_logic_r0_baseline_csv_on`
- Run 1: `artifacts/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_on/20260423T195838_math_logic_r1_baseline_csv_on`

Conclusion: heartbeat-on SAAQ 1.5 validation completed successfully on OLMoE with matched repeat determinism.