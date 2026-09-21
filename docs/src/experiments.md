# SNN Market-Microstructure Experiments

DendriteTrader is an experimental market-microstructure and paper-execution
laboratory. It evaluates whether spike-based models produce useful, calibrated
forecasts from causally available market events after declared latency, fill, fee,
and risk assumptions.

It is not a production HFT runtime, an authenticated exchange client, or a source of
financial advice. Venue-critical execution remains outside this Julia package.

## Experiment boundary

The research pipeline is intentionally decomposed:

```text
market events -> causal features -> spike encoder -> forecast model
              -> calibration -> policy -> paper simulator -> evidence artifacts
```

DendriteTrader owns finance-specific schemas, features, labels, policies, paper
execution, and experiment orchestration. Reusable neuron dynamics, plasticity, and
generic SNN training belong in dedicated neuromorphic packages and connect through
adapters. Deterministic low-latency IPC and real venue execution belong in separate
services.

## Scientific requirements

Every reported experiment must preserve:

- a chronological train/validation/test split with an embargo at least as long as
  the maximum feature or forecast horizon;
- training-only normalization and calibration;
- explicit event-time and physical-time horizons;
- non-SNN baselines evaluated on the same decisions and market events;
- declared fees, latency, fill, and queue assumptions;
- per-session metrics and aggregate uncertainty, not only one headline total;
- configuration, data hashes, code revision, seeds, environment, and model identity;
- negative and null findings;
- no credentials, wallet material, or live order submission.

## Evidence families

Experiments report four complementary groups of measurements:

1. Predictive: MCC, macro F1, balanced accuracy, and per-class precision/recall.
2. Calibration: Brier score, expected calibration error, and reliability bins.
3. Paper execution: net PnL, implementation shortfall, turnover, fill rate,
   inventory exposure, drawdown, and completed-transaction correctness.
4. Systems/SNN: encoding and inference latency percentiles, throughput, spike
   density, silent or saturated channels, memory, and a clearly labeled energy proxy
   or hardware measurement.

Classification performance alone is not evidence of an actionable strategy.

## Initial campaign

The first campaign compares a queue-imbalance rule, a linear classifier, a
uniform-time-constant LIF network, and a mixed-memory LIF network. All models consume
the same causal book-derived features and are evaluated at the same decision times.

The campaign starts with deterministic synthetic fixtures in CI. Real-data runs use
separately acquired data with a session manifest recording license, venue,
instrument, timestamp semantics, depth, parser revision, raw hashes, sequence-gap
policy, and known limitations. Raw licensed or large market data is not committed.

Zero-cost and zero-latency scenarios are diagnostics, not realistic claims. L2 data
may support an explicitly approximate queue model; queue-accurate claims require
suitable order-level data.

