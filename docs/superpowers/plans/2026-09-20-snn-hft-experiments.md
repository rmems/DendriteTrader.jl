# SNN HFT Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, leakage-resistant market-microstructure experiment pipeline that compares SNNs with ordinary baselines under identical paper-execution assumptions.

**Architecture:** DendriteTrader owns finance-domain events, causal features, labels, policies, simulation, and experiment artifacts. Generic neuron/training implementations remain external and enter through model adapters; production IPC and authenticated venue execution remain out of scope.

**Tech Stack:** Julia 1.10+, Test stdlib, JSON, TOML stdlib, existing HTTP/ZMQ control-plane adapters.

**Spec:** `docs/src/experiments.md`

## Global Constraints

- Paper trading and research only; no authenticated exchange order placement.
- Preserve the existing `TradeSignal` and ZMQ interfaces while introducing research-native contracts additively.
- Store prices and sizes as integers in canonical replay types.
- Every rolling transform consumes only observations at or before its decision timestamp.
- Fit normalization, calibration, thresholds, and model selection on train/validation data only.
- Commit small fixtures and manifests, never licensed or large raw market data.
- Every commit includes `Co-authored-by: Codex <noreply@openai.com>`.

---

### Task 1: Correctness foundation

**Files:**
- Modify: `src/DendriteTrader.jl`
- Modify: `src/backtest/Backtest.jl`
- Modify: `test/runtests.jl`
- Create: `docs/src/experiments.md`

**Interfaces:**
- Produces: `latency_ns(signal, observed_ns)::Int64` for deterministic replay.
- Produces: signed quantities in `ExecutionEngine.positions`.
- Produces: an equity curve marked to the latest observation after every signal.

- [x] **Step 1: Add failing deterministic-latency tests.**
- [x] **Step 2: Add failing signed-short accounting test.**
- [x] **Step 3: Add failing interim mark-to-market and drawdown tests.**
- [x] **Step 4: Implement the minimal latency, signed-position, and equity changes.**
- [x] **Step 5: Run `julia --startup-file=no --project=. test/runtests.jl` and confirm all tests pass.**

### Task 2: Canonical market-event replay

**Files:**
- Create: `src/market/Market.jl`
- Create: `src/market/Events.jl`
- Create: `src/market/OrderBook.jl`
- Create: `src/market/Replay.jl`
- Create: `test/market_replay.jl`
- Create: `test/fixtures/book_session.jsonl`
- Modify: `src/DendriteTrader.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `BookDelta`, `TradePrint`, `BookSnapshot`, `OrderBookState`, `ReplaySession`, `replay!`, and `load_session_jsonl`.
- Invariant: events are strictly ordered by `(exchange_ts_ns, sequence)` and sequence gaps are rejected unless the session policy explicitly allows them.

- [ ] **Step 1: Write fixture tests for add, reduce, delete, best bid/ask, spread, and depth snapshots.**
- [ ] **Step 2: Run the replay tests and verify the market types are undefined.**
- [ ] **Step 3: Implement integer-tick events and deterministic L2 book mutation.**
- [ ] **Step 4: Add tests for duplicate sequences, gaps, crossed books, and out-of-order timestamps.**
- [ ] **Step 5: Implement explicit replay validation errors and session policy.**
- [ ] **Step 6: Run the full suite and commit the replay kernel.**

### Task 3: Causal features, labels, and chronological splits

**Files:**
- Create: `src/features/Features.jl`
- Create: `src/features/Microstructure.jl`
- Create: `src/features/Normalization.jl`
- Create: `src/features/Labels.jl`
- Create: `src/experiments/Splits.jl`
- Create: `test/features.jl`
- Create: `test/leakage.jl`
- Modify: `src/DendriteTrader.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `FeatureFrame`, `FeatureRow`, `RollingZScore`, `fit!`, `transform!`, `MovementLabel`, `label_event_horizon`, `ChronologicalSplit`, and `walk_forward_splits`.
- Consumes: ordered `BookSnapshot` values from Task 2.

- [ ] **Step 1: Write hand-derived tests for spread, mid-price, microprice, imbalance, and signed order flow.**
- [ ] **Step 2: Verify tests fail because the feature module is absent.**
- [ ] **Step 3: Implement the five causal features without future access.**
- [ ] **Step 4: Write leakage-canary tests proving validation/test values cannot affect training normalization.**
- [ ] **Step 5: Implement training-only rolling normalization and serializable parameters.**
- [ ] **Step 6: Write exact-boundary tests for event-horizon labels, session boundaries, and embargoes.**
- [ ] **Step 7: Implement three-way movement labels and walk-forward split validation.**
- [ ] **Step 8: Run the full suite and commit the causal dataset layer.**

### Task 4: Forecast, encoder, and baseline contracts

**Files:**
- Create: `src/models/Models.jl`
- Create: `src/models/Interface.jl`
- Create: `src/models/Baselines.jl`
- Create: `src/spikes/Spikes.jl`
- Create: `src/spikes/Interface.jl`
- Create: `src/spikes/DeltaEncoder.jl`
- Create: `test/models.jl`
- Create: `test/spikes.jl`
- Modify: `src/DendriteTrader.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `ForecastHorizon`, `Forecast`, `AbstractForecastModel`, `fit!`, `predict!`, `reset_state!`, `AbstractSpikeEncoder`, `SpikeFrame`, and `encode!`.
- Produces baselines: `StationaryModel`, `ImbalanceRule`, and `RidgeClassifier`.

- [ ] **Step 1: Write tests for forecast probability validation and deterministic model reset.**
- [ ] **Step 2: Implement the forecast/model protocol.**
- [ ] **Step 3: Write literal-fixture tests for stationary, imbalance, and ridge baselines.**
- [ ] **Step 4: Implement the baselines with an injected RNG where stochastic behavior exists.**
- [ ] **Step 5: Write delta-encoder tests for positive, negative, silent, and saturated channels.**
- [ ] **Step 6: Implement delta encoding and spike-density diagnostics.**
- [ ] **Step 7: Run the full suite and commit the model boundary.**

### Task 5: Event-driven paper simulator

**Files:**
- Create: `src/simulation/Simulation.jl`
- Create: `src/simulation/Orders.jl`
- Create: `src/simulation/Latency.jl`
- Create: `src/simulation/Fills.jl`
- Create: `src/simulation/Portfolio.jl`
- Create: `src/simulation/Metrics.jl`
- Create: `test/simulation.jl`
- Modify: `src/DendriteTrader.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `OrderIntent`, `PaperOrder`, `Fill`, `FixedLatency`, `L2QueueApproximation`, `PaperPortfolio`, `simulate!`, and `SimulationResult`.
- Consumes: forecasts from Task 4 and replay events from Task 2.

- [ ] **Step 1: Write tests proving no fill occurs before decision time plus latency.**
- [ ] **Step 2: Implement fixed-latency order activation.**
- [ ] **Step 3: Write tests for spread crossing, passive non-fill, partial fill, cancellation, fees, and rebates.**
- [ ] **Step 4: Implement deterministic seeded fills with assumptions stored in the result.**
- [ ] **Step 5: Write inventory, cash, realized PnL, unrealized PnL, and drawdown reconciliation tests.**
- [ ] **Step 6: Implement the signed portfolio ledger and per-event mark-to-market curve.**
- [ ] **Step 7: Run the full suite and commit the simulator.**

### Task 6: Reproducible experiment artifacts

**Files:**
- Create: `src/experiments/Experiments.jl`
- Create: `src/experiments/Config.jl`
- Create: `src/experiments/Runner.jl`
- Create: `src/experiments/Artifacts.jl`
- Create: `experiments/configs/001_lif_vs_imbalance.toml`
- Create: `scripts/run_experiment.jl`
- Create: `test/experiments.jl`
- Modify: `src/DendriteTrader.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `ExperimentConfig`, `ExperimentManifest`, `ExperimentResult`, `run_experiment`, and `write_artifacts`.
- Artifacts: `manifest.json`, `metrics.json`, `predictions.jsonl`, and `summary.md`.

- [ ] **Step 1: Write deterministic artifact tests with literal expected hashes for the tiny fixture.**
- [ ] **Step 2: Implement canonical JSON field ordering and atomic artifact writes.**
- [ ] **Step 3: Write config validation tests for overlapping splits, insufficient embargo, undeclared seed, and zero-cost claim labeling.**
- [ ] **Step 4: Implement TOML configuration loading and validation.**
- [ ] **Step 5: Run the baseline and SNN-adapter candidates through one common runner.**
- [ ] **Step 6: Record code revision, data hashes, environment, seeds, timings, assumptions, and null findings in the manifest.**
- [ ] **Step 7: Run the full suite and the synthetic smoke experiment, then commit the experiment runner.**

### Task 7: Shadow-data adapter and systems benchmark

**Files:**
- Create: `src/control_plane/ShadowFeed.jl`
- Create: `scripts/run_shadow.jl`
- Create: `scripts/benchmark_inference.jl`
- Create: `test/shadow_feed.jl`
- Modify: `docs/src/experiments.md`
- Modify: `README.md`

**Interfaces:**
- Produces: an unauthenticated market-data-to-hypothetical-order path and JSON systems benchmark.
- Constraint: the adapter has no order-submission method and accepts no wallet or API secret.

- [ ] **Step 1: Write a fixture-driven test proving the adapter emits hypothetical orders only.**
- [ ] **Step 2: Implement the shadow feed behind the existing read-only client boundary.**
- [ ] **Step 3: Write benchmark-output tests for p50, p95, p99, throughput, and spike density.**
- [ ] **Step 4: Implement the benchmark with warmup and measured iterations declared in output.**
- [ ] **Step 5: Document reproduction, limitations, and the explicit no-live-trading boundary.**
- [ ] **Step 6: Run tests, docs, formatting, and the synthetic end-to-end campaign.**

