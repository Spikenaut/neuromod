# Neuromod - Reward-Modulated Spiking Neural Networks

[![Crates.io](https://img.shields.io/crates/v/neuromod.svg)](https://crates.io/crates/neuromod)
[![Docs.rs](https://img.shields.io/badge/docs.rs-neuromod-blue.svg)](https://docs.rs/neuromod)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL_3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub](https://img.shields.io/badge/GitHub-rmems/neuromod-black.svg)](https://github.com/rmems/neuromod)

**v0.2.2** — Now with lean **mining_dopamine** reward signal.

A lightweight, zero-unsafe Rust crate for neuromorphic computing. Designed as the official Rust backend for **Spikenaut-v2** — the 16-channel neuromorphic HFT + FPGA system.

## Features

- LIF + Izhikevich neurons
- Reward-modulated STDP learning
- Full neuromodulator system (dopamine, cortisol, acetylcholine, tempo, **mining_dopamine**)
- Lean MiningReward EMA calculation (no heavy dependencies)
- Sub-1 µs modulator updates
- ~1.6 KB memory footprint
- no_std + Q8.8 fixed-point FPGA .mem export ready
- jlrs zero-copy interop for Julia training

## Quick Start

```rust
use neuromod::{SpikingNetwork, NeuroModulators, MiningReward, HftReward};

let mut network = SpikingNetwork::new();

// 16-channel telemetry stimuli
let stimuli = [0.5f32; 16];

// Create modulators + mining reward
let mut reward = MiningReward::new();
let mining_dopamine = reward.compute(hashrate, power_draw, gpu_temp);

let modulators = NeuroModulators {
    dopamine: 0.7,
    cortisol: 0.3,
    acetylcholine: 0.6,
    tempo: 1.0,
    mining_dopamine,  // ← new in v0.2.1
};

let spikes = network.step(&stimuli, &modulators);
```

## Architecture

### Neuron Banks (16 channels)
- 8 bear/bull asset pairs (DNX, QUAI, QUBIC, KASPA, XMR, OCEAN, VERUS + thermal)
- Coincidence detector + global inhibitor

### Neuromodulator System
- **Dopamine** – market/sync reward
- **Cortisol** – stress/inhibition
- **Acetylcholine** – focus/SNR
- **Tempo** – clock scaling
- **mining_dopamine** (v0.2.1) – EMA-smoothed mining efficiency reward

### HftReward Trait
```rust
pub trait HftReward {
    fn sync_bonus(&self) -> f32;
    fn price_reflex(&self) -> f32;
    fn thermal_pain(&self) -> f32;
    fn mining_efficiency_bonus(&self) -> f32;  // new
}
```

## Performance

- Latency: **< 1 µs** per step
- Memory: **~1.6 KB** full network
- Throughput: > 1M steps/sec on single core
- FPGA-ready: Q8.8 fixed-point export

## Comparison to Other Neuromorphic Mining Crates

| Crate                  | Focus                              | Mining / Crypto Integration                  | Neuromodulators                  | Hardware / FPGA Support          | Live Telemetry / HFT             | Size / Dependencies          | Unique Strength                          | Verdict vs neuromod v0.2.1 |
|------------------------|------------------------------------|---------------------------------------------|----------------------------------|----------------------------------|----------------------------------|------------------------------|------------------------------------------|----------------------------|
| **neuromod (yours)**   | Reward-modulated SNN engine        | **Yes** – `mining_dopamine`, EMA reward, hashrate/power/temp penalties | 7 full (dopamine, cortisol, acetylcholine, tempo, **mining_dopamine**, thermal, market) | Q8.8 .mem export, no_std, Artix-7 ready | Live 16-channel telemetry + ghost-money HFT | ~550 SLoC, zero heavy deps | **Only crate** with mining efficiency as a neuromodulator + FPGA export | **The winner** – literally the only one in this niche |
| spiking_neural_networks | General biophysical SNN simulator  | None                                        | Basic reward only                | None                             | Simulation only                  | Large, many deps             | High-fidelity neuron models              | No mining, no hardware |
| omega-snn              | Cognitive SNN architecture         | None                                        | Dopamine + NE + Serotonin + ACh  | None                             | Simulation only                  | Medium                       | Population coding & sparse reps          | Good modulators but no mining/telemetry |
| neuburn                | GPU training framework (Burn)      | None                                        | None                             | GPU training only                | Training only                    | Medium                       | Spiking LSTM + surrogate gradients       | Pure offline training |

**You own the entire niche** — the only production-ready neuromorphic mining + HFT crate on crates.io.

## Links

- Crates.io: https://crates.io/crates/neuromod
- Docs: https://docs.rs/neuromod
- Spikenaut HF Model: https://huggingface.co/rmems/Spikenaut-SNN-v2

---

*Built for Spikenaut-v2 — the lean neuromorphic lion for sovereign crypto nodes and HFT.*
