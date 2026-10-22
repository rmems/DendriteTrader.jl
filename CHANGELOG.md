# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking: sizing rejects invalid inputs instead of clamping)

Research invariant 7: unsupported or invalid sizes and prices are rejected with an
explicit reason, never silently clamped or coerced. Existing keyword calls keep
working, but the public API changes: `ExecutionEngine`, the `BacktestConfig` keyword
constructor and `load_config` accept a new `max_position_fraction`; `BacktestConfig`
gains a ninth field (its full positional constructor takes nine arguments, and an
eight-argument form is kept); `DEFAULT_MAX_POSITION_FRACTION` is a new export; and the
`max_position_size` default changes from `10.0` to `Inf`.

- `execute_signal!` no longer caps Kelly-sized units with `min(units, max_position_size)`.
  A signal whose Kelly size exceeds a position cap is rejected: `executed = false`,
  `rejected_signals` is incremented, a `max_position_reject` event is recorded, and the
  reason names the cap (`exceeds_max_position:` or `exceeds_max_position_fraction:`).
  Sizes exactly at a cap are accepted.
- **Default position cap is now relative.** New `max_position_fraction` on
  `ExecutionEngine`, `BacktestConfig` and `load_config` (default
  `DEFAULT_MAX_POSITION_FRACTION = 1.0`): an order is rejected when the projected
  position in the ticker, `abs(current_units ± units) * price`, exceeds
  `max_position_fraction * account_balance` (the `account_balance` passed to
  `execute_signal!`; `run_backtest` passes its cash balance, which does not fall when a position opens (only commissions and realized PnL change it)). Orders that only reduce the existing
  position in the ticker are exempt; a flip past flat is checked. The cap is per ticker
  and does not bound total notional across tickers (`max_portfolio_exposure` does). `max_position_size` becomes an
  optional absolute unit cap with default `Inf` (was `10.0`); when both are set, both
  apply. The fixed 10-unit default rejected ordinary entries (a `10_000` balance at
  price `100` sizes ~46 units at confidence `0.95`), so a default backtest made zero
  trades. Why `1.0`: `execute_signal!` sizes with `size_position` at
  `kelly_scalar = 0.5`, which commits at most `0.5` of the balance; the `[0, 0.20]`
  output cap belongs to `kelly_fraction`, which this path does not use, so a relative
  default such as `0.25` would still reject ordinary entries. `1.0` never binds on a
  single Kelly-sized entry from flat; it binds when same-side entries accumulate (a
  third Kelly buy at confidence `0.92` is rejected) and on oversized explicit quantities.
- `ExecutionEngine` and `BacktestConfig` throw `ArgumentError` for a `max_position_fraction`
  that is `NaN`, `±Inf` or `<= 0`, and for a `max_position_size` that is `NaN` or `<= 0`
  (`Inf` means no absolute cap). An explicit `position_units_override` above either cap
  throws `ArgumentError` carrying the cap reason. `BacktestConfig` keeps an eight-argument
  positional constructor that uses the default relative cap.
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
  and floored at `1e-9`). It also throws when the derived ratio `avg_win / avg_loss`
  overflows or underflows (previously the result was `NaN`). The `[0, 0.20]` output cap is
  unchanged documented policy. `size_position` likewise throws if the derived unit count
  overflows. `kelly_scalar = 0` stays valid and yields a zero-unit position (`zero_reject`).
- `half_kelly` (and `from_confidence`, which delegates to it) throws `ArgumentError` when
  `p` is outside `[0, 1]` or non-finite, or `b` is non-positive or non-finite (previously
  clamped and floored at `1e-9`). The `[0, 1]` output cap is unchanged.
- A `NaN` `max_position_size` previously disabled the hard cap, because `units > NaN`
  is false; it now throws. `load_config` already rejected non-positive values from files.
- `run_backtest` inherits the engine change: an entry above a configured cap is rejected
  rather than opened at the cap. Pass `max_position_size` to restore an absolute unit cap.

### Added

- **Latency-aware paper simulator (RM-1864)**: `simulate!` replays a `ReplaySession` with separate information, decision, activation and fill clocks (`FixedLatency`, `ReceiveTimestampFeed` / `FixedFeedDelay`), an explicitly approximate `L2QueueApproximation` fill model, integer `FeeSchedule`, bounded `RiskLimits`, and a per-step reconciled `PaperPortfolio` evaluation state. `SimulationResult` stores all assumptions, seeds and a SHA-256 result hash. Vendored `metabolic-ledger.parity.v1` fixtures are replayed with `run_parity_fixture`; `parity_document` emits long/flat runs in the same schema. Paper only; no order submission. Runs with any zero order-entry, cancel or report latency are labelled `diagnostic_zero_latency` in their assumptions and hash (`is_zero_latency_diagnostic`).
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