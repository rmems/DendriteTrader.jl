# DendriteTrader

[![Docs](https://github.com/Limen-Neural/DendriteTrader.jl/actions/workflows/docs.yml/badge.svg)](https://limen-neural.github.io/DendriteTrader.jl)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](https://github.com/Limen-Neural/DendriteTrader.jl/blob/main/LICENSE)

Julia strategy, diagnostics, paper-trading, and control-plane tooling for neural trading systems.

DendriteTrader consumes neural trade signals, applies confidence gating, sizes positions with integrated Kelly/fractional-Kelly helpers, tracks paper positions, and exposes read-only market-data utilities. It is intentionally scoped as the Julia-side control-plane layer; deterministic low-latency execution loops belong in Rust services such as `corpus-ipc` and adjacent ledger infrastructure such as `metabolic-ledger`.

## Features

- **ZMQ SUB socket** — consumes JSON trade signals from Rust or other signal producers
- **Nanosecond latency tracking** — Unix-epoch timestamps aligned with Rust `timestamp_nanos`
- **Confidence gate** — only accepts signals above a configurable threshold
- **Integrated Kelly sizing** — `kelly_fraction`, `half_kelly`, `from_confidence`, and `size_position`
- **Paper position tracking** — updates in-memory positions for strategy/control-plane decisions
- **Portfolio risk cap** — optional gross-units cap across names (`max_portfolio_exposure`, `portfolio_risk`)
- **dYdX v4 market data** — read-only REST client for orderbook and oracle price queries

## Repository Boundary

DendriteTrader is for Julia-side strategy and control-plane responsibilities:

- Strategy-facing signal ingestion and diagnostics
- Confidence gating and position sizing
- Paper-trading state transitions
- Read-only market data helpers
- Human-readable experimentation and test coverage around neural confidence signals

DendriteTrader is **not** the deterministic HFT execution runtime. Latency-critical production execution, IPC loops, and venue-critical paths should remain in Rust services, including `corpus-ipc`.

**Ownership split with `metabolic-ledger`:**

| Repo | Owns |
|------|------|
| **DendriteTrader.jl** | `TradeSignal`, confidence gate, Kelly sizing proposals (`kelly_fraction`, `from_confidence`, `PositionSize`), dYdX v4 REST client, ZMQ SUB consumer — **signal → decision only** |
| **metabolic-ledger** | `GhostWallet`, buy/sell state transitions, ATP energy, `GhostTradeLog`, realized PnL, win/loss tracking — **persistent accounting** |

See the module docstring in `src/DendriteTrader.jl` for the explicit boundary declaration.

## Ecosystem

DendriteTrader is designed to remain composable with the broader Julia finance ecosystem rather than hard-code a single asset class or venue. Packages such as `Miletus.jl` can model complex financial contracts, while `LimitOrderBook.jl` can support high-frequency limit order book simulations. DendriteTrader should focus on neural signal control-plane logic, paper-trading decisions, and sizing helpers that can be manually adapted to whichever instruments, contracts, books, or venues are used later.

## Architecture

```text
Julia strategy/control-plane        Rust infrastructure             Exchange / ledger
        ↓                                  ↑                              ↑
SNN signal diagnostics → confidence gate → Kelly sizing → paper decision → corpus-ipc / metabolic-ledger
        │                                                                  │
        └────────────── read-only dYdX market data helpers ────────────────┘
```

## Installation

```julia
] instantiate
```

Or add directly:

```julia
] add https://github.com/Limen-Neural/DendriteTrader.jl
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `ZMQ`   | ZeroMQ SUB socket for signal ingestion |
| `HTTP`  | REST calls to dYdX v4 indexer |
| `JSON`  | Signal deserialization |

## Quick Start

### 1. Create an execution engine

```julia
using DendriteTrader

engine = ExecutionEngine(
    confidence_threshold     = Float32(0.85),
    max_position_size        = 10.0,
    max_portfolio_exposure   = Inf,
    payoff_ratio             = 1.5,
    log_file                 = "events.jsonl",
)
```

### 2. Process a signal manually

```julia
signal = TradeSignal(Dict(
    "ticker"       => "MARKET-PAIR",
    "side"         => "BUY",
    "price"        => 100.0,
    "quantity"     => 1.0,
    "confidence"   => 0.92,
    "timestamp_ns" => time_ns(),
))

decision = execute_signal!(engine, signal, 10_000.0)
println(decision)
```

### 3. Persist and replay events (optional)

```julia
# Each SignalEvent is appended as a JSON line with a schema_version field
# Close the log handle when finished (also flushed by stop!)
close_log!(engine)

# Load the history back for audit or replay
history = load_history("events.jsonl")
```

### 4. Start the ZMQ listener

```julia
start!(
    engine;
    zmq_endpoint = "tcp://localhost:5555",
    on_decision = decision -> begin
        if decision.executed
            println("Executed: $(decision.signal.ticker) × $(round(decision.position_units, digits=4))")
            println("  Latency: $(decision.latency_ns) ns")
            println("  Kelly fraction: $(round(decision.kelly_fraction, digits=4))")
        else
            println("Rejected: $(decision.reason)")
        end
    end,
)
```

### 5. Fetch market data for any supported venue symbol

```julia
client = DydxClient(base_url = "https://indexer.dydx.trade/v4")

price = get_price(client, "MARKET-PAIR")
if price !== nothing
    println("Mid:  \$$(mid_price(price))")
    println("Spread: $(spread_bps(price)) bps")
end
```

## Kelly Sizing

DendriteTrader folds the former standalone SpikeKelly sizing helpers into the main package. The sizing API maps neural confidence scores and trade statistics to Kelly or fractional-Kelly capital allocations.

```julia
using DendriteTrader

fraction = kelly_fraction(win_rate = 0.55, avg_win = 8.50, avg_loss = 5.20)
half = half_kelly(0.60, 1.5)
confidence_fraction = from_confidence(confidence = 0.90, payoff_ratio = 1.5)

position = size_position(
    confidence = 0.90,
    price = 100.0,
    account_balance = 10_000.0,
    payoff_ratio = 1.5,
)

println(position.units)
println(position.kelly_fraction)
println(position.risk)
```

### Sizing API

| Type / Function | Description |
|-----------------|-------------|
| `kelly_fraction(; win_rate, avg_win, avg_loss)` | Conservative half-Kelly fraction from historical trade statistics, clamped to `[0.0, 0.20]` |
| `half_kelly(p, b)` | Half-Kelly fraction for win probability `p` and payoff ratio `b` |
| `from_confidence(; confidence, payoff_ratio)` | Maps neural confidence directly to a half-Kelly fraction |
| `RiskTier` | Qualitative risk enum: `Aggressive`, `Moderate`, `Conservative`, `Minimal` |
| `risk_tier(confidence)` | Classifies confidence into a `RiskTier` |
| `PositionSize` | Result struct with units, Kelly fraction, confidence, risk tier, and account risk percentage |
| `size_position(; confidence, price, account_balance, payoff_ratio, kelly_scalar)` | Computes full position sizing from neural confidence |

## Rate Limiter

Token bucket rate limiter for controlling API call rates. Thread-safe via `ReentrantLock`. Integrates with `DydxClient` to throttle REST requests.

```julia
using DendriteTrader

# Create a limiter: 10 requests/sec, burst of 10
limiter = RateLimiter(requests_per_second = 10.0, burst = 10.0)

# Block until a token is available
acquire!(limiter)

# Update rate at runtime
set_rate!(limiter, 20.0)
```

### RateLimiter API

| Type / Function | Description |
|-----------------|-------------|
| `RateLimiter(; requests_per_second, burst)` | Create a token bucket limiter. `requests_per_second` sets the refill rate; `burst` sets the max token capacity. Both must be `> 0`. |
| `acquire!(limiter)` | Block until a token is available. Refills tokens based on elapsed time. Thread-safe. |
| `set_rate!(limiter, requests_per_second)` | Update the refill rate at runtime. Thread-safe. |

### Integrating RateLimiter with DydxClient

```julia
limiter = RateLimiter(requests_per_second = 5.0, burst = 5.0)
client = DydxClient(
    base_url = "https://indexer.dydx.trade/v4",
    rate_limit = limiter,
)
```

## Price Cache

Short TTL cache for dYdX price data. Avoids redundant REST calls when prices are queried frequently. Thread-safe via `ReentrantLock`.

```julia
using DendriteTrader

# Create a cache with 5-second TTL
cache = PriceCache(ttl_s = 5.0)

# Manually store a price
price = DydxPrice("BTC-USD", 65_000.0, 64_990.0, 65_010.0)
put_cached!(cache, "BTC-USD", price)

# Retrieve if fresh (returns nothing if expired)
cached = get_cached(cache, "BTC-USD")

# Check freshness
is_fresh(cache, "BTC-USD")  # true if within TTL

# Invalidate a single ticker
invalidate!(cache, "BTC-USD")

# Clear all entries
clear!(cache)

# Current entry count
cache_size(cache)
```

### PriceCache API

| Type / Function | Description |
|-----------------|-------------|
| `PriceCache(; ttl_s)` | Create a price cache with TTL in seconds (default 5.0). |
| `get_cached(cache, ticker)` | Return cached `DydxPrice` if fresh, `nothing` otherwise. |
| `put_cached!(cache, ticker, price)` | Store a price with current timestamp. |
| `invalidate!(cache, ticker)` | Remove a specific ticker from the cache. |
| `clear!(cache)` | Remove all entries from the cache. |
| `cache_size(cache)` | Return the number of cached entries. |
| `is_fresh(cache, ticker)` | Return `true` if the cached entry is within TTL. |

### Integrating PriceCache with DydxClient

```julia
cache = PriceCache(ttl_s = 10.0)
client = DydxClient(
    base_url = "https://indexer.dydx.trade/v4",
    cache = cache,
)

# get_price will check cache first, then fetch via REST
price = get_price(client, "BTC-USD")
```

## Backtest

Paper-trading backtest harness. Replays historical signals through the `ExecutionEngine` and computes performance metrics (total return, max drawdown, win rate). Supports loading signals from JSON or CSV and exporting results.

```julia
using DendriteTrader

# Configure
config = BacktestConfig(
    initial_balance = 10_000.0,
    confidence_threshold = 0.85,
    payoff_ratio = 1.5,
)

# Load historical signals
signals = load_signals_json("data/signals.json")
# or: signals = load_signals_csv("data/signals.csv")

# Run backtest
result = run_backtest(config, signals)

# Print summary table
print_summary(result)

# Persist backtest event trail by passing log_file to BacktestConfig or run_backtest
config_log = BacktestConfig(
    initial_balance = 10_000.0,
    log_file = "backtest_events.jsonl",
)
result = run_backtest(config_log, signals)

# Export results
export_equity_csv(result, "output/equity.csv")
export_trade_log_json(result, "output/trades.json")
```

### Backtest API

| Type / Function | Description |
|-----------------|-------------|
| `BacktestConfig(; initial_balance, confidence_threshold, payoff_ratio, max_position_size, max_portfolio_exposure, risk_free_rate, slippage_pct, commission_pct, log_file)` | Configuration for a backtest run. Defaults: balance `$10,000`, threshold `0.85`, payoff `1.5`, max units `10.0`, portfolio cap `Inf`, risk-free rate `0.0`, slippage `0.0%`, commission `0.0%`. Optional `log_file` persists `SignalEvent`s. |
| `run_backtest(config, signals; log_file=config.log_file)` | Replay `Vector{TradeSignal}` through the engine. Returns a `BacktestResult`. |
| `BacktestResult` | Result struct with `config`, `initial_balance`, `final_balance`, `equity_curve`, `trade_log`, `events`, `total_return`, `max_drawdown`, `win_rate`, `total_trades`. |
| `print_summary(result)` | Print a formatted summary table to stdout. |
| `load_signals_json(path)` | Load signals from a JSON file (array of signal dicts). |
| `load_signals_csv(path)` | Load signals from a CSV file. Expected columns: `ticker, side, price, quantity, confidence, timestamp_ns`. |
| `export_equity_csv(result, path)` | Export the equity curve to CSV (`index,equity`). |
| `export_trade_log_json(result, path)` | Export the trade log to JSON. |

### BacktestResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `config` | `BacktestConfig` | Configuration used for the backtest run |
| `initial_balance` | `Float64` | Starting account balance |
| `final_balance` | `Float64` | Ending account balance |
| `total_return` | `Float64` | Total return percentage |
| `max_drawdown` | `Float64` | Maximum drawdown percentage |
| `win_rate` | `Float64` | Fraction of profitable trades (of closed trades) |
| `total_trades` | `Int` | Number of trades executed |
| `equity_curve` | `Vector{Float64}` | Balance after each signal |
| `trade_log` | `Vector{TradeRecord}` | All executed trades with PnL |
| `events` | `Vector{SignalEvent}` | Raw engine events |

## Signal Format

The ZMQ listener expects JSON objects matching this schema:

```json
{
  "ticker":       "MARKET-PAIR",
  "side":         "BUY",
  "price":        100.0,
  "quantity":     1.0,
  "confidence":   0.92,
  "timestamp_ns": 1744156800000000000
}
```

| Field | Type | Description |
|-------|------|-------------|
| `ticker` | string | Configurable venue symbol, e.g. `"MARKET-PAIR"` |
| `side` | string | `"BUY"`, `"SELL"`, or `"NEUTRAL"` |
| `price` | float | Expected execution price in USD |
| `quantity` | float | Producer-provided units; DendriteTrader may compute its own Kelly-sized units |
| `confidence` | float | Neural model output in `[0.0, 1.0]` |
| `timestamp_ns` | int | Unix nanoseconds at signal creation |

## API Reference

### Core Types

| Type / Function | Description |
|-----------------|-------------|
| `TradeSignal` | Immutable struct holding a deserialized trade signal |
| `ExecutionEngine` | Stateful engine with confidence gate, Kelly sizing, and position tracking. Optional `log_file` appends events as JSON lines. |
| `ExecutionDecision` | Result of processing one signal, including requested `kelly_fraction` and capped `applied_fraction` |
| `execute_signal!(engine, signal, balance)` | Gate, size, and process a single signal |
| `latency_ns(signal)` | End-to-end latency in nanoseconds |
| `passes_gate(signal, threshold)` | Boolean check: `signal.confidence >= threshold` |
| `fill_rate(engine)` | Fraction of signals executed vs. rejected |
| `load_history(path)` | Load a JSON-lines `SignalEvent` history; returns `SignalEvent[]` for missing/empty files and skips malformed lines |
| `close_log!(engine)` | Flush and close the event log file handle |

### dYdX v4 Client

| Type / Function | Description |
|-----------------|-------------|
| `DydxClient` | REST client for read-only dYdX v4 market data |
| `get_price(client, ticker)` | Fetch oracle price and orderbook top as `DydxPrice` or `nothing` |
| `DydxPrice` | Struct with `oracle_price`, `best_bid`, and `best_ask` |
| `mid_price(p)` | Mid between best bid and ask |
| `spread_bps(p)` | Bid-ask spread in basis points |

## Migration Notes

### From SpikenautExecution

The package/module identity is now `DendriteTrader`.

```julia
# Before
using SpikenautExecution

# After
using DendriteTrader
```

The repository is scoped as a Julia strategy/control-plane, diagnostics, and paper-trading package rather than a low-latency production execution loop.

### From SpikeKelly

The standalone SpikeKelly package is superseded by DendriteTrader's integrated sizing API.

```julia
# Before
using SpikenautKelly

# After
using DendriteTrader
```

The former SpikeKelly function names are preserved inside DendriteTrader where possible: `kelly_fraction`, `half_kelly`, `from_confidence`, `risk_tier`, and `size_position`.

## Running Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

Dual-licensed under MIT or Apache-2.0.

- See [`LICENSE-MIT`](LICENSE-MIT)
- See [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0)
- Or [`LICENSE`](LICENSE) for the dual-license declaration.
