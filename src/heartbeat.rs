//! Deterministic heartbeat-style perturbations for telemetry validation runs.

use crate::types::{HeartbeatConfig, TelemetrySnapshot};

#[derive(Debug, Clone)]
pub struct HeartbeatInjector {
    config: HeartbeatConfig,
}

impl HeartbeatInjector {
    pub fn new(config: HeartbeatConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &HeartbeatConfig {
        &self.config
    }

    pub fn signal_at_tick(&self, tick: usize) -> f32 {
        if !self.config.enabled || self.config.amplitude.abs() <= f32::EPSILON {
            return 0.0;
        }

        let period = self.config.period_ticks.max(1);
        let duty_cycle = self.config.duty_cycle.clamp(0.05, 0.95);
        let phase = ((tick + self.config.phase_offset_ticks) % period) as f32 / period as f32;
        let pulse = if phase < duty_cycle {
            let attack = (phase / duty_cycle).clamp(0.0, 1.0);
            1.0 - attack * 0.15
        } else {
            let recovery = ((phase - duty_cycle) / (1.0 - duty_cycle)).clamp(0.0, 1.0);
            -0.35 * recovery
        };

        pulse * self.config.amplitude
    }

    pub fn apply(&self, base: &TelemetrySnapshot, tick: usize) -> TelemetrySnapshot {
        let signal = self.signal_at_tick(tick);
        let mut snapshot = base.clone();
        snapshot.gpu_temp_c = (snapshot.gpu_temp_c + signal * 6.0).max(0.0);
        snapshot.gpu_power_w = (snapshot.gpu_power_w + signal * 42.0).max(0.0);
        snapshot.cpu_tctl_c = (snapshot.cpu_tctl_c + signal * 4.5).max(0.0);
        snapshot.cpu_package_power_w = (snapshot.cpu_package_power_w + signal * 24.0).max(0.0);
        snapshot.heartbeat_signal = signal;
        snapshot.heartbeat_enabled = self.config.enabled;
        snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_snapshot() -> TelemetrySnapshot {
        TelemetrySnapshot {
            gpu_temp_c: 70.0,
            gpu_power_w: 240.0,
            cpu_tctl_c: 72.0,
            cpu_package_power_w: 115.0,
            heartbeat_signal: 0.0,
            heartbeat_enabled: false,
            timestamp_ms: 0,
        }
    }

    #[test]
    fn disabled_heartbeat_keeps_snapshot_unchanged() {
        let injector = HeartbeatInjector::new(HeartbeatConfig::default());
        let snap = injector.apply(&base_snapshot(), 12);
        assert_eq!(snap.gpu_temp_c, 70.0);
        assert_eq!(snap.heartbeat_signal, 0.0);
        assert!(!snap.heartbeat_enabled);
    }

    #[test]
    fn heartbeat_signal_is_deterministic() {
        let injector = HeartbeatInjector::new(HeartbeatConfig {
            enabled: true,
            amplitude: 0.8,
            period_ticks: 16,
            duty_cycle: 0.25,
            phase_offset_ticks: 3,
        });
        assert_eq!(injector.signal_at_tick(5), injector.signal_at_tick(5));
        assert_ne!(injector.signal_at_tick(5), injector.signal_at_tick(13));
    }
}
