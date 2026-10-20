# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Leakage canaries for splits and labels** (`test/leakage_splits_labels.jl`): walk-forward splits checked by event sequence on feature frames with skipped events and on row tables with several rows per event; `RidgeClassifier` parameters shown independent of rows and labels past the training frame; structural check that `MovementTarget` cannot be passed to `predict!`, `encode!`, `transform!`, sizing, or signal-execution APIs. Each canary has a deliberately leaky control. Two `@test_broken` assertions document that `walk_forward_splits` embargoes in row positions, not event sequences, so an event-denominated horizon under-embargoes when several rows share an event.
- `CLAUDE.md`: Claude Code notes importing `AGENTS.md`, with research invariants, known invariant gaps, verified commands, and an architecture map.
- **Event persistence**: `ExecutionEngine` can append every `SignalEvent` to a JSON-lines log (`log_file` and `truncate` kwargs). New `load_history(path)` reads the log back, skipping missing/empty/malformed lines. New `close_log!(engine)` flushes and closes the log handle.
- **Portfolio exposure cap**: `ExecutionEngine` accepts `max_portfolio_exposure` (default `Inf`, unlimited), a cap on total gross notional exposure (`Σ |quantity| × last execution price`) across all tickers. `execute_signal!` rejects signals whose resulting exposure would exceed the cap (`portfolio_reject` event); signals that only reduce an existing position are always allowed. New `portfolio_risk(engine)` accessor returns the current gross notional exposure. `load_config` supports the `max_portfolio_exposure` key (must be positive).
- `SignalEvent` now captures the decision outcome with `executed`, `position_units`, and `applied_fraction` fields, and includes a `schema_version` field in its JSON representation.
- `BacktestConfig` and `run_backtest` accept an optional `log_file` to persist backtest event trails.

### Changed

- **CI format gate**: the `JuliaFormatter` CI job now actually fails on unformatted code. It previously ran `format(".", check=true)`, which JuliaFormatter 2.x ignores (it reformatted in place and exited 0). The job now installs JuliaFormatter `2.x` into a temporary environment and exits 1 when `format("."; overwrite=false)` returns `false`. `AGENTS.md` documents the same command. `src/features/Labels.jl`, `src/market/Replay.jl` and `test/leakage.jl` were reformatted (layout only, parsed ASTs unchanged).
- License: switched from GPL-3.0-only to dual MIT OR Apache-2.0.
  - Added `LICENSE-MIT` and `LICENSE-APACHE-2.0`.
  - Updated top-level `LICENSE` to declare dual licensing.
  - Updated README.md License section and added SPDX identifiers to source.
  - Added `license` field to `Project.toml` with SPDX expression.
  - This change is for maximum adoption across research, hardware, BCI, HFT, and bindings.

See issues #10 and #11.

## [0.1.0] - 2026-07-05

### Added

- **ZMQ signal ingestion**: ZeroMQ-based signal receiver for real-time trading signals.
- **Confidence gating**: Minimum confidence threshold to filter low-quality signals.
- **Kelly sizing**: Kelly criterion-based position sizing for optimal capital allocation.
- **dYdX v4 REST client**: REST API client for dYdX v4 decentralized exchange integration.
- **Paper position tracking**: Simulated position tracking for paper trading without live execution.

[0.1.0]: https://github.com/Limen-Neural/DendriteTrader.jl/releases/tag/v0.1.0