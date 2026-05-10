# Model onboarding: <provider/model>

## Official source
- Provider:
- Model ID:
- Model card URL:
- License/access:
- Verified official provider? yes/no

## Model structure
- Dense or MoE:
- Active parameters:
- Total parameters:
- Experts:
- Context length:
- Checkpoint format:
- Quantization format:

## Target environment
- Local / cloud:
- Preferred provider:
- GPU target:
- Minimum VRAM assumption:
- Storage requirement:

## SAAQ scope
- SAAQ version:
- Telemetry source:
- Heartbeat mode:
- Tensor/synapse target:
- Expected output artifacts:

## Acceptance criteria
- [ ] Model source verified
- [ ] License/access checked
- [ ] Model added to lineup config
- [ ] Dry-run path validation passes
- [ ] SAAQ smoke run completes
- [ ] `run_manifest.json` emitted
- [ ] `artifacts/index.csv` updated
- [ ] Sentry/New Relic fields populated where enabled
- [ ] README/docs updated

## Non-goals
- No full fine-tuning
- No SAAQ math changes
- No router math changes
- No unsafe path/secret logging