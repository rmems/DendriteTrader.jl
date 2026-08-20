# Design: multi-asset portfolio risk cap (GH#72 / LIM-796)

Date: 2026-08-20

## Problem

`ExecutionEngine` already clamps each fill to `max_position_size` (units). Nothing caps **total** exposure across names, so many assets can each sit at the per-name cap.

## Decision

Measure exposure as **gross units**: `sum(abs(qty))` over `engine.positions`. Positions today never go negative (sells clamp at 0), so this is the sum of open quantities.

Do **not** clip fills. If a signal’s projected gross units would **exceed** the cap, reject it. Equal-to-cap is allowed.

Default `max_portfolio_exposure = Inf` so existing callers and tests are unchanged until a finite cap is set.

## Engine

New field / constructor kwarg: `max_portfolio_exposure::Float64` (default `Inf`).

`execute_signal!` order stays: confidence gate → Neutral reject → Kelly size → per-name `min(units, max_position_size)` → zero-size reject → **portfolio check** → update book.

Projected gross:

- Buy: `current_gross + units`
- Sell: `current_gross - min(units, current_qty_for_ticker)`

Reject path: increment `rejected_signals`, record `SignalEvent` with `event_type = "portfolio_reject"`, return `ExecutionDecision` with `executed = false` and a reason that includes projected exposure and cap. Positions are not mutated.

## Accessor

```julia
portfolio_risk(engine) -> NamedTuple{(:exposure, :cap, :utilization), ...}
```

- `exposure`: current gross units
- `cap`: `max_portfolio_exposure`
- `utilization`: `exposure / cap` when `cap` is finite and `> 0`, else `0.0`

## Backtest

`BacktestConfig` gains the same kwarg (default `Inf`) and `run_backtest` forwards it to `ExecutionEngine`, matching `max_position_size`.

## Tests

- Default cap is `Inf`; `portfolio_risk` utilization is 0
- Two names under a finite cap: first buy executes, second that would breach rejects and leaves the book unchanged
- Sell that reduces gross still executes when already at the cap
- Per-name cap still applies first (a fill is never larger than `max_position_size` even when portfolio room remains)

## Out of scope

Notional / account-fraction caps, short inventory, clipping leftover capacity, persistent ledger state (`metabolic-ledger`).
