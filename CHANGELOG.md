# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking: sizing rejects invalid inputs instead of clamping)

Research invariant 7: unsupported or invalid sizes and prices are rejected with an
explicit reason, never silently clamped or coerced. Public signatures are unchanged.

- `execute_signal!` no longer caps Kelly-sized units with `min(units, max_position_size)`.
  A signal whose Kelly size exceeds `max_position_size` is rejected: `executed = false`,
  `rejected_signals` is incremented, a `max_position_reject` event is recorded, and the
  reason starts with `exceeds_max_position:`. Units exactly at the cap are accepted.
  With the defaults (`max_position_size = 10.0`, `payoff_ratio = 1.5`,
  `kelly_scalar = 0.5`) and a `10_000` balance, the Kelly size exceeds 10 units
  whenever the price is below about `1_000 × k` (for example below ~433 at
  confidence 0.92, where `k ≈ 0.433`), so those signals are now rejected. Raise
  `max_position_size` to execute them at their full Kelly size.
- `execute_signal!` rejects invalid sizing inputs (`price <= 0`, `account_balance <= 0`,
  non-finite values, confidence outside `[0, 1]`) with a `sizing_reject` event and a reason
  starting with `invalid_sizing_input:`. Previously these were a `zero_reject` with the
  generic reason `"zero-sized position"`. An explicit `position_units_override` (used by
  `run_backtest` to close a position) still proceeds when the balance is finite but
  `<= 0`, because it does not use Kelly sizing; the price is still validated.
- `size_position` throws `ArgumentError` when `price <= 0`, `account_balance <= 0`, either
  is non-finite, `confidence` is outside `[0, 1]` or non-finite, `payoff_ratio <= 0`, or
  `kelly_scalar < 0`. Previously it returned a zero-unit `PositionSize` with no reason for
  non-positive price/balance, clamped `confidence` into `[0, 1]`, and floored
  `payoff_ratio` at `1e-9`. A negative edge still returns a zero-unit position.
- `kelly_fraction` throws `ArgumentError` when `win_rate` is outside `[0, 1]` or
  non-finite, or `avg_win`/`avg_loss` is non-positive or non-finite (previously clamped
  and floored at `1e-9`). The `[0, 0.20]` output cap is unchanged documented policy.
- `half_kelly` (and `from_confidence`, which delegates to it) throws `ArgumentError` when
  `p` is outside `[0, 1]` or non-finite, or `b` is non-positive or non-finite (previously
  clamped and floored at `1e-9`). The `[0, 1]` output cap is unchanged.
- `run_backtest` inherits the engine change: with the default `BacktestConfig`
  (`max_position_size = 10.0`), entries whose Kelly size exceeds 10 units are now
  rejected rather than opened at 10 units.

### Added

- **Event persistence**: `ExecutionEngine` can append every `SignalEvent` to a JSON-lines log (`log_file` and `truncate` kwargs). New `load_history(path)` reads the log back, skipping missing/empty/malformed lines. New `close_log!(engine)` flushes and closes the log handle.
- **Portfolio exposure cap**: `ExecutionEngine` accepts `max_portfolio_exposure` (default `Inf`, unlimited), a cap on total gross notional exposure (`Σ |quantity| × last execution price`) across all tickers. `execute_signal!` rejects signals whose resulting exposure would exceed the cap (`portfolio_reject` event); signals that only reduce an existing position are always allowed. New `portfolio_risk(engine)` accessor returns the current gross notional exposure. `load_config` supports the `max_portfolio_exposure` key (must be positive).
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