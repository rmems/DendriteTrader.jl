# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Portfolio risk cap**: `ExecutionEngine` accepts `max_portfolio_exposure` (gross units across names, default `Inf`). Signals that would exceed the cap are rejected. New `portfolio_risk(engine)` reports current exposure, cap, and utilization. `BacktestConfig` forwards the same knob.
- **Event persistence**: `ExecutionEngine` can append every `SignalEvent` to a JSON-lines log (`log_file` and `truncate` kwargs). New `load_history(path)` reads the log back, skipping missing/empty/malformed lines. New `close_log!(engine)` flushes and closes the log handle.
- `SignalEvent` now captures the decision outcome with `executed`, `position_units`, and `applied_fraction` fields, and includes a `schema_version` field in its JSON representation.
- `BacktestConfig` and `run_backtest` accept an optional `log_file` to persist backtest event trails.

### Changed

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