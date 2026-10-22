# DendriteTrader.jl

[![CI](https://github.com/Limen-Neural/DendriteTrader.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Limen-Neural/DendriteTrader.jl/actions/workflows/ci.yml)
[![Docs](https://github.com/Limen-Neural/DendriteTrader.jl/actions/workflows/docs.yml/badge.svg)](https://limen-neural.github.io/DendriteTrader.jl)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](https://github.com/Limen-Neural/DendriteTrader.jl/blob/main/LICENSE)

Julia strategy, diagnostics, paper-trading, and control-plane layer for neural trading systems.

DendriteTrader consumes neural trade signals, applies confidence gating, sizes positions with
Kelly/fractional-Kelly helpers, tracks paper positions, and exposes read-only market-data utilities.
It is the Julia-side control-plane; deterministic low-latency execution loops belong in adjacent
Rust services.

## Architecture

```text
Julia strategy/control-plane        Rust infrastructure             Exchange / ledger
        ↓                                  ↑                              ↑
SNN signal diagnostics → confidence gate → Kelly sizing → paper decision → corpus-ipc / metabolic-ledger
        │                                                                  │
        └────────────── read-only dYdX market data helpers ────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| [`DendriteTrader`](@ref) | Core execution engine, ZMQ listener, dYdX REST client, price cache, rate limiter |
| [`SizingModule`](@ref DendriteTrader.SizingModule) | Kelly and fractional-Kelly position sizing |
| [`Backtest`](@ref DendriteTrader.Backtest) | Paper-trading backtest harness with performance metrics |

## Quick Start

```julia
using DendriteTrader

engine = ExecutionEngine(
    confidence_threshold = Float32(0.85),
    max_position_fraction = 1.0,   # default: per-ticker position notional ≤ 1.0 × account_balance
    payoff_ratio         = 1.5,
)

signal = TradeSignal(Dict(
    "ticker"       => "MARKET-PAIR",
    "side"         => "BUY",
    "price"        => 100.0,
    "quantity"     => 1.0,
    "confidence"   => 0.92,
    "timestamp_ns" => round(Int64, time() * 1e9),
))

decision = execute_signal!(engine, signal, 10_000.0)
# Kelly sizing proposes ~43 units here (k ≈ 0.433 of the balance), within the
# relative cap, so it executes at full size. An order above a cap is rejected
# ("exceeds_max_position_fraction: ..." or "exceeds_max_position: ..."), never shrunk.
println(decision.executed, " — ", decision.reason)
```

## Installation

```julia
] add https://github.com/Limen-Neural/DendriteTrader.jl
```

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE](https://github.com/Limen-Neural/DendriteTrader.jl/blob/main/LICENSE).
