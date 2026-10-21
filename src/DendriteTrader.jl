# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    DendriteTrader

Julia strategy, diagnostics, paper-trading, and control-plane layer for neural trading systems.

Bridges Julia SNN strategy output to trading decisions with:
- ZMQ SUB socket for JSON trade signals from Rust nervous system
- Nanosecond latency tracking (signal creation → execution)
- Confidence gate (only executes signals above threshold)
- Integrated Kelly/fractional-Kelly position sizing
- dYdX v4 decentralized perpetuals REST client (no API key required)

## Architecture

```
Julia (Brain)          Rust (Muscle)           Exchange
     ↓                      ↑
LIF neurons → SNN signal → ZMQ bridge → Kelly sizing → dydx v4 → Order
(16 neurons)  (confidence)   (IPC)        (sizing)     (REST)
```

## Provenance

Developed as a Julia control-plane and paper-trading package for custom neuromorphic SNN models and ML trading integrations.
Latency-critical execution loops remain the responsibility of adjacent Rust services.

## Repository Boundary

DendriteTrader owns:
- `TradeSignal`, confidence gating, `ExecutionDecision`
- Kelly sizing proposals (`kelly_fraction`, `from_confidence`, `PositionSize`, `size_position`)
- dYdX v4 REST client (read-only market data)
- ZMQ SUB consumer — signal → decision only

DendriteTrader does **NOT** own:
- `GhostWallet`-like persistent accounting state
- Win rate, realized PnL, portfolio position tracking

Win-rate/PnL tracking belongs in `metabolic-ledger`.

## References

- Kelly, J.L. (1956). A New Interpretation of Information Rate.
  *Bell System Technical Journal*, 35(4), 917–926.
  https://doi.org/10.1002/j.1538-7305.1956.tb03809.x
  Position sizing via the Kelly Criterion.

- dYdX Foundation (2024). *dYdX v4 Indexer API Documentation*.
  REST endpoints for orderbook queries and perpetual market data.
  https://docs.dydx.exchange/api_integration-indexer/indexer_api

- iMatix Corporation (2013). *ZeroMQ: Messaging for Many Applications*.
  O'Reilly Media. SUB socket pattern for trade signal delivery.
  https://zguide.zeromq.org

## Usage

```julia
using DendriteTrader

engine = ExecutionEngine(confidence_threshold=0.85)
start!(engine, zmq_endpoint="tcp://localhost:5555")
```
"""
module DendriteTrader

using JSON
using HTTP
using TOML
using YAML
using ZMQ

export TradeSignal,
    TradeSide,
    Buy,
    Sell,
    Neutral,
    ExecutionEngine,
    ExecutionDecision,
    SignalEvent,
    load_config,
    load_history,
    close_log!
export validate_signal, execute_signal!, latency_ns, passes_gate
export DydxClient, DydxPrice, get_price, mid_price, spread_bps
export RateLimiter, acquire!, set_rate!
export PriceCache, invalidate!, clear!, cache_size, get_cached, put_cached!, is_fresh
export start!, stop!, events, fill_rate, portfolio_risk, DEFAULT_MAX_POSITION_FRACTION
export kelly_fraction, from_confidence, half_kelly
export RiskTier, Aggressive, Moderate, Conservative, Minimal, risk_tier
export PositionSize, size_position
export BacktestConfig, BacktestResult, run_backtest, print_summary
export load_signals_json, load_signals_csv, export_equity_csv, export_trade_log_json

include("sizing/SizingModule.jl")
using .SizingModule

# ── Trade Signal ─────────────────────────────────────────────────────────────

"""
    @enum TradeSide

Direction of a trade signal.
"""
@enum TradeSide begin
    Buy = 1
    Sell = 2
    Neutral = 0
end

include("market/Market.jl")
using .Market

export BookSide, Bid, Ask
export MarketEvent, BookDelta, TradePrint, BookSnapshot
export OrderBookState, ReplayPolicy, ReplaySession
export apply!, snapshot, best_bid, best_ask, spread_ticks, replay!, load_session_jsonl

include("features/Features.jl")
using .Features

export FeatureRow, FeatureFrame, microstructure_features
export RollingZScore, fit!, transform!, normalization_parameters
export MovementLabel, MovementTarget, Down, Flat, Up, label_event_horizon

include("experiments/Experiments.jl")
using .Experiments

export ChronologicalSplit, walk_forward_splits

"""Reset transient state in an encoder or forecast model without discarding fitted parameters."""
function reset_state! end

include("models/Models.jl")
using .Models

export ForecastHorizon, Forecast, AbstractForecastModel, predict!, reset_state!
export StationaryModel, ImbalanceRule, RidgeClassifier

include("spikes/Spikes.jl")
using .Spikes

export AbstractSpikeEncoder, SpikeFrame, DeltaEncoder, encode!, spike_density

include("simulation/Simulation.jl")
using .Simulation

export OrderIntent, CancelIntent, RiskLimits, PaperOrder, OpenOrderView
export FixedLatency, AbstractFeedDelay, ReceiveTimestampFeed, FixedFeedDelay
export activation_time, cancel_effective_time, available_time
export FeeSchedule, fill_fee, L2QueueApproximation, Fill
export ParityScale, PortfolioError, PaperPortfolio, AuditRow, apply_fill!, record_non_fill!
export notional_subunits, unrealized_pnl, weighted_cost_basis_ticks, in_parity_subset, reconciles
export SimulationConfig, DecisionContext, DecisionRecord, LedgerPoint, SimulationResult
export SimulationMetrics, simulate!, ForecastPolicy, reconciliation_errors
export zero_latency_components, is_zero_latency_diagnostic
export decision_records, fill_records, audit_dict
export PARITY_SCHEMA_ID, load_parity_fixture, run_parity_fixture, parity_document

"""
    TradeSignal

Deserialized trade signal from Rust nervous system via ZMQ.

# Fields
- `ticker`:       configurable venue symbol
- `side`:         Buy / Sell / Neutral
- `price`:        expected execution price (USD)
- `quantity`:     units to trade (Kelly-sized by Rust, override in Julia)
- `confidence`:   SNN output [0.0, 1.0]
- `timestamp_ns`: Unix nanoseconds for latency tracking
"""
struct TradeSignal
    ticker::String
    side::TradeSide
    price::Float64
    quantity::Float64
    confidence::Float32
    timestamp_ns::Int64
end

"""
    validate_signal(d::AbstractDict) -> Union{Nothing, String}

Validate a trade signal dictionary. Returns `nothing` if valid, or an error message string.

# Required Fields
- `ticker`: String, non-empty
- `side`: "BUY", "SELL", or "NEUTRAL"
- `price`: Number > 0
- `confidence`: Number in [0.0, 1.0]
- `timestamp_ns`: Integer > 0
"""
function validate_signal(d::AbstractDict)
    # Check required fields
    for field in ("ticker", "side", "price", "confidence", "timestamp_ns")
        if !haskey(d, field)
            return "missing required field: $field"
        end
    end

    # Validate ticker
    ticker = d["ticker"]
    if !(ticker isa AbstractString) || isempty(ticker)
        return "ticker must be a non-empty string"
    end

    # Validate side
    side = d["side"]
    if !(side isa AbstractString) || !(side in ("BUY", "SELL", "NEUTRAL"))
        return "side must be BUY, SELL, or NEUTRAL, got: $side"
    end

    # Validate price
    price = d["price"]
    if price isa Bool || !(price isa Real)
        return "price must be a number, got: $(typeof(price))"
    end
    if !isfinite(Float64(price))
        return "price must be finite, got: $price"
    end
    if price <= 0
        return "price must be positive, got: $price"
    end

    # Validate confidence
    confidence = d["confidence"]
    if confidence isa Bool || !(confidence isa Real)
        return "confidence must be a number, got: $(typeof(confidence))"
    end
    if !isfinite(Float64(confidence))
        return "confidence must be finite, got: $confidence"
    end
    if confidence < 0.0 || confidence > 1.0
        return "confidence must be in [0.0, 1.0], got: $confidence"
    end

    # Validate timestamp_ns
    timestamp = d["timestamp_ns"]
    if timestamp isa Bool || !(timestamp isa Integer)
        return "timestamp_ns must be an integer, got: $(typeof(timestamp))"
    end
    if timestamp <= 0
        return "timestamp_ns must be positive, got: $timestamp"
    end

    return nothing
end

function TradeSignal(d::AbstractDict)
    side = if get(d, "side", "NEUTRAL") == "BUY"
        Buy
    elseif get(d, "side", "NEUTRAL") == "SELL"
        Sell
    else
        Neutral
    end
    TradeSignal(
        get(d, "ticker", "UNKNOWN"),
        side,
        Float64(get(d, "price", 0.0)),
        Float64(get(d, "quantity", 0.0)),
        Float32(get(d, "confidence", 0.0)),
        Int64(get(d, "timestamp_ns", 0)),
    )
end

"""
    latency_ns(signal) -> Int64
    latency_ns(signal, observed_ns) -> Int64

End-to-end latency from signal creation to an observation timestamp. The one-argument
method uses the current Unix epoch time for live messages. Pass `observed_ns` during
deterministic replay so results do not depend on the wall clock.
"""
function latency_ns(s::TradeSignal)::Int64
    now_ns = round(Int64, time() * 1_000_000_000)
    return latency_ns(s, now_ns)
end

latency_ns(s::TradeSignal, observed_ns::Integer)::Int64 =
    observed_ns <= s.timestamp_ns ? Int64(0) : Int64(observed_ns - s.timestamp_ns)

"""
    passes_gate(signal, threshold) -> Bool

True if signal confidence meets or exceeds the threshold.
"""
passes_gate(s::TradeSignal, threshold::Float32) = s.confidence >= threshold

# ── Execution Decision ────────────────────────────────────────────────────────

"""
    ExecutionDecision

Result of processing one signal through the execution engine.
"""
struct ExecutionDecision
    signal::TradeSignal
    executed::Bool
    reason::String
    kelly_fraction::Float64
    applied_fraction::Float64
    position_units::Float64
    latency_ns::Int64
end

# ── Structured Logging ────────────────────────────────────────────────────────

"""
    SignalEvent

Structured event for signal lifecycle tracking, including the resulting decision.
"""
struct SignalEvent
    event_type::String
    ticker::String
    confidence::Float32
    side::String
    reason::String
    kelly_fraction::Float64
    latency_ns::Int64
    timestamp::Float64
    executed::Bool
    position_units::Float64
    applied_fraction::Float64
end

"""
    SignalEvent(event_type, ticker, confidence, side; reason="", kelly_fraction=0.0, latency_ns=0, executed=false, position_units=0.0, applied_fraction=0.0)

Convenience constructor for SignalEvent with default values.
"""
function SignalEvent(
    event_type::String,
    ticker::String,
    confidence::Float32,
    side::String;
    reason::String = "",
    kelly_fraction::Float64 = 0.0,
    latency_ns::Int64 = 0,
    executed::Bool = false,
    position_units::Float64 = 0.0,
    applied_fraction::Float64 = 0.0,
)
    SignalEvent(
        event_type,
        ticker,
        confidence,
        side,
        reason,
        kelly_fraction,
        latency_ns,
        time(),
        executed,
        position_units,
        applied_fraction,
    )
end

"""
    SignalEvent(event_type, ticker, confidence, side, reason, kelly_fraction, latency_ns, timestamp)

Backward-compatible 8-argument positional constructor.
"""
function SignalEvent(
    event_type::String,
    ticker::String,
    confidence::Float32,
    side::String,
    reason::String,
    kelly_fraction::Float64,
    latency_ns::Int64,
    timestamp::Float64,
)
    SignalEvent(
        event_type,
        ticker,
        confidence,
        side,
        reason,
        kelly_fraction,
        latency_ns,
        timestamp,
        false,
        0.0,
        0.0,
    )
end

"""
    SignalEvent(d::AbstractDict)

Deserialize a SignalEvent from a dictionary (typically parsed from a JSON line).
"""
function SignalEvent(d::AbstractDict)
    executed = get(d, "executed", false)
    if executed isa AbstractString
        executed = parse(Bool, executed)
    end
    SignalEvent(
        d["event_type"],
        d["ticker"],
        Float32(d["confidence"]),
        d["side"],
        get(d, "reason", ""),
        Float64(get(d, "kelly_fraction", 0.0)),
        Int64(get(d, "latency_ns", 0)),
        Float64(get(d, "timestamp", time())),
        executed,
        Float64(get(d, "position_units", 0.0)),
        Float64(get(d, "applied_fraction", 0.0)),
    )
end

const _EVENT_SCHEMA_VERSION = 1

function _event_to_dict(event::SignalEvent)
    Dict(
        "schema_version" => _EVENT_SCHEMA_VERSION,
        "event_type" => event.event_type,
        "ticker" => event.ticker,
        "confidence" => event.confidence,
        "side" => event.side,
        "reason" => event.reason,
        "kelly_fraction" => event.kelly_fraction,
        "latency_ns" => event.latency_ns,
        "timestamp" => event.timestamp,
        "executed" => event.executed,
        "position_units" => event.position_units,
        "applied_fraction" => event.applied_fraction,
    )
end

# ── Execution Engine ──────────────────────────────────────────────────────────

const _EXECUTION_CONFIG_KEYS = Set((
    "confidence_threshold",
    "max_position_size",
    "max_position_fraction",
    "max_portfolio_exposure",
    "payoff_ratio",
    "log_file",
    "truncate",
))

function _config_number(config::AbstractDict, key::String, target::Type)
    value = config[key]
    value isa Real && !(value isa Bool) ||
        throw(ArgumentError("configuration key `$key` must be a number, got $(repr(value))"))
    converted = try
        target(value)
    catch err
        if err isa ArgumentError || err isa InexactError || err isa OverflowError
            throw(
                ArgumentError(
                    "configuration key `$key` must be convertible to $target, got $(repr(value))",
                ),
            )
        end
        rethrow()
    end
    isfinite(converted) ||
        throw(ArgumentError("configuration key `$key` must be finite, got $(repr(value))"))
    return converted
end

"""
    load_config(path) -> ExecutionEngine

Create an [`ExecutionEngine`](@ref) from a flat TOML (`.toml`) or YAML
(`.yaml`/`.yml`) file. The supported keys are `confidence_threshold`,
`max_position_size`, `max_position_fraction`, `max_portfolio_exposure`,
`payoff_ratio`, `log_file`, and `truncate`; omitted keys use the constructor
defaults. Numeric configuration values are converted to the engine's concrete
numeric types, and YAML `null` is accepted for `log_file`. All numeric values must
be finite. Omit `max_position_size` for no absolute unit cap and
`max_portfolio_exposure` for the unlimited exposure default (both `Inf`); when set
in a file each must be a positive finite cap. `confidence_threshold` must be in
`[0, 1]`, and `max_position_size`, `max_position_fraction`,
`max_portfolio_exposure`, and `payoff_ratio` must be positive. Relative
`log_file` paths are resolved relative to the configuration file.

An unsupported extension, a non-mapping YAML document, an unknown key, or a
value of the wrong type raises `ArgumentError`. Syntax errors from the selected
parser are propagated.
"""
function load_config(path::AbstractString)
    extension = lowercase(splitext(path)[2])
    config = if extension == ".toml"
        TOML.parsefile(path)
    elseif extension == ".yaml" || extension == ".yml"
        YAML.load_file(path)
    else
        throw(ArgumentError("unsupported configuration format `$extension`; use .toml, .yaml, or .yml"))
    end

    config isa AbstractDict || throw(ArgumentError("configuration root must be a mapping"))
    for key in keys(config)
        key isa AbstractString ||
            throw(ArgumentError("configuration keys must be strings, got $(repr(key))"))
        key in _EXECUTION_CONFIG_KEYS || throw(ArgumentError("unknown configuration key `$key`"))
    end

    kwargs = Pair{Symbol, Any}[]
    if haskey(config, "confidence_threshold")
        confidence_threshold = _config_number(config, "confidence_threshold", Float32)
        0.0f0 <= confidence_threshold <= 1.0f0 ||
            throw(ArgumentError("configuration key `confidence_threshold` must be between 0 and 1"))
        push!(kwargs, :confidence_threshold => confidence_threshold)
    end
    if haskey(config, "max_position_size")
        max_position_size = _config_number(config, "max_position_size", Float64)
        max_position_size > 0.0 ||
            throw(ArgumentError("configuration key `max_position_size` must be positive"))
        push!(kwargs, :max_position_size => max_position_size)
    end
    if haskey(config, "max_position_fraction")
        max_position_fraction = _config_number(config, "max_position_fraction", Float64)
        max_position_fraction > 0.0 ||
            throw(ArgumentError("configuration key `max_position_fraction` must be positive"))
        push!(kwargs, :max_position_fraction => max_position_fraction)
    end
    if haskey(config, "max_portfolio_exposure")
        max_portfolio_exposure = _config_number(config, "max_portfolio_exposure", Float64)
        max_portfolio_exposure > 0.0 ||
            throw(ArgumentError("configuration key `max_portfolio_exposure` must be positive"))
        push!(kwargs, :max_portfolio_exposure => max_portfolio_exposure)
    end
    if haskey(config, "payoff_ratio")
        payoff_ratio = _config_number(config, "payoff_ratio", Float64)
        payoff_ratio > 0.0 ||
            throw(ArgumentError("configuration key `payoff_ratio` must be positive"))
        push!(kwargs, :payoff_ratio => payoff_ratio)
    end
    if haskey(config, "log_file")
        log_file = config["log_file"]
        (log_file isa AbstractString || isnothing(log_file)) || throw(
            ArgumentError(
                "configuration key `log_file` must be a string or null, got $(repr(log_file))",
            ),
        )
        resolved_log_file = if isnothing(log_file)
            nothing
        elseif isabspath(log_file)
            String(log_file)
        else
            normpath(joinpath(dirname(abspath(path)), log_file))
        end
        push!(kwargs, :log_file => resolved_log_file)
    end
    if haskey(config, "truncate")
        truncate = config["truncate"]
        truncate isa Bool || throw(
            ArgumentError("configuration key `truncate` must be a Bool, got $(repr(truncate))"),
        )
        push!(kwargs, :truncate => truncate)
    end

    return ExecutionEngine(; kwargs...)
end

"""
    ExecutionEngine

Stateful execution engine with confidence gating and position management.

# Fields
- `confidence_threshold`: minimum SNN confidence to execute (default 0.85)
- `max_position_size`:    optional absolute cap on order units (default `Inf`, no
                          absolute cap). Must be positive; `NaN`, `0` and negative
                          values throw `ArgumentError`. Kelly-sized units above the
                          cap are rejected (`max_position_reject`, reason prefix
                          `exceeds_max_position:`), not shrunk to the cap.
- `max_position_fraction`: relative cap on order notional as a fraction of the
                          account balance (default `1.0`).
                          An order is rejected when
                          `units * price > max_position_fraction * account_balance`
                          (`max_position_reject`, reason prefix
                          `exceeds_max_position_fraction:`). Must be finite and
                          positive. Orders that only reduce the existing position in
                          that ticker are exempt, because both the price and the
                          balance can drift after entry and a cap must not trap a
                          position. See [`DEFAULT_MAX_POSITION_FRACTION`](@ref) for
                          why the default never binds on Kelly-sized entries. When
                          both caps are set, both apply.
- `max_portfolio_exposure`: cap on total gross notional exposure in price
                          units across all tickers (default `Inf`, unlimited).
                          Positions are valued at the last execution price seen
                          for each ticker; a signal that only reduces an
                          existing position is never rejected by this cap.
- `payoff_ratio`:         odds-style average win/loss ratio for Kelly sizing (default 1.5)
- `positions`:            current open positions (ticker → quantity)
- `last_prices`:          last execution price per ticker, for exposure valuation
- `log_file`:             optional path to a JSON-lines event log
- `log_io`:               open file handle for the event log (if configured)
- `log_lock`:             lock serializing `log_io` writes
"""
mutable struct ExecutionEngine
    confidence_threshold::Float32
    max_position_size::Float64
    max_position_fraction::Float64
    max_portfolio_exposure::Float64
    payoff_ratio::Float64
    positions::Dict{String, Float64}
    last_prices::Dict{String, Float64}
    total_signals::Int
    executed_signals::Int
    rejected_signals::Int
    should_stop::Threads.Atomic{Bool}
    events::Vector{SignalEvent}
    log_file::Union{String, Nothing}
    log_io::Union{IOStream, Nothing}
    log_lock::ReentrantLock
end

"""
    close_log!(engine)

Flush and close the event log file handle, if one is open.
"""
function close_log!(engine::ExecutionEngine)
    lock(engine.log_lock) do
        if engine.log_io !== nothing
            try
                flush(engine.log_io)
                close(engine.log_io)
            catch
            end
            engine.log_io = nothing
        end
    end
    return engine
end

"""
    flush_log!(engine)

Flush the event log file handle, if one is open.
"""
function flush_log!(engine::ExecutionEngine)
    lock(engine.log_lock) do
        if engine.log_io !== nothing && isopen(engine.log_io)
            flush(engine.log_io)
        end
    end
    return engine
end

function _write_event!(engine::ExecutionEngine, event::SignalEvent)
    if engine.log_io === nothing
        return event
    end
    lock(engine.log_lock) do
        if engine.log_io !== nothing && isopen(engine.log_io)
            println(engine.log_io, JSON.json(_event_to_dict(event)))
        end
    end
    return event
end

function _record_event!(engine::ExecutionEngine, event::SignalEvent)
    push!(engine.events, event)
    _write_event!(engine, event)
    return event
end

"""
    DEFAULT_MAX_POSITION_FRACTION = 1.0

Default relative position cap for [`ExecutionEngine`](@ref) and `BacktestConfig`: an
order's notional (`units * price`) may not exceed this fraction of the account
balance, i.e. no leverage.

Why `1.0`: `execute_signal!` sizes with [`size_position`](@ref) at its default
`kelly_scalar = 0.5`, whose Kelly fraction is `0.5 * (p*b - q)/b`, capped to `[0, 1]`.
Because `(p*b - q)/b <= 1` for every `p` in `[0, 1]` and `b > 0`, a Kelly-sized entry
commits at most `0.5` of the balance (reached only at confidence `1.0`). A default of
`1.0` therefore never rejects a Kelly-sized entry and leaves a 2x margin, so
floating-point rounding at the boundary cannot flip a decision. It is a backstop
against unlevered-notional violations (for example an explicit
`position_units_override`), not a sizing policy.

The `[0, 0.20]` output cap of [`kelly_fraction`](@ref) does not apply here:
`execute_signal!` does not call `kelly_fraction`, so Kelly-sized entries can commit up
to `0.5` of the balance (about `0.458` at confidence `0.95` with `payoff_ratio = 1.5`).
A relative cap at or below `0.5`, such as `0.25`, would reject ordinary high-confidence
entries.
"""
const DEFAULT_MAX_POSITION_FRACTION = 1.0

# Shared constructor validation for ExecutionEngine and BacktestConfig. Caps are never
# coerced: a NaN cap would make `x > cap` false and silently disable it.
function _validate_position_caps(max_position_size::Real, max_position_fraction::Real)
    (!isnan(max_position_size) && max_position_size > 0.0) || throw(
        ArgumentError(
            "max_position_size must be positive (Inf means no absolute cap), got $(repr(max_position_size))",
        ),
    )
    (isfinite(max_position_fraction) && max_position_fraction > 0.0) || throw(
        ArgumentError(
            "max_position_fraction must be finite and positive, got $(repr(max_position_fraction))",
        ),
    )
    return nothing
end

"""
    ExecutionEngine(; confidence_threshold=0.85f0, max_position_size=Inf, max_position_fraction=DEFAULT_MAX_POSITION_FRACTION, max_portfolio_exposure=Inf, payoff_ratio=1.5, log_file=nothing, truncate=false)

Create an execution engine. If `log_file` is provided, events are appended as JSON-lines.
Pass `truncate=true` to truncate an existing log.

`max_position_size` is an optional absolute unit cap (`Inf` means none) and
`max_position_fraction` a relative notional cap (`units * price` over the account
balance). Both apply when both are set. Invalid caps (`NaN`, non-positive, or a
non-finite fraction) throw `ArgumentError`.
"""
function ExecutionEngine(;
    confidence_threshold::Float32 = Float32(0.85),
    max_position_size::Float64 = Inf,
    max_position_fraction::Float64 = DEFAULT_MAX_POSITION_FRACTION,
    max_portfolio_exposure::Float64 = Inf,
    payoff_ratio::Float64 = 1.5,
    log_file::Union{String, Nothing} = nothing,
    truncate::Bool = false,
)
    _validate_position_caps(max_position_size, max_position_fraction)
    (isfinite(max_portfolio_exposure) || isinf(max_portfolio_exposure)) &&
    max_portfolio_exposure > 0.0 || throw(
        ArgumentError(
            "max_portfolio_exposure must be positive (Inf means unlimited), got $(repr(max_portfolio_exposure))",
        ),
    )
    log_io = if log_file === nothing
        nothing
    else
        dir = dirname(log_file)
        if !isempty(dir)
            mkpath(dir)
        end
        open(log_file, truncate ? "w" : "a")
    end

    engine = ExecutionEngine(
        confidence_threshold,
        max_position_size,
        max_position_fraction,
        max_portfolio_exposure,
        payoff_ratio,
        Dict{String, Float64}(),
        Dict{String, Float64}(),
        0,
        0,
        0,
        Threads.Atomic{Bool}(false),
        SignalEvent[],
        log_file,
        log_io,
        ReentrantLock(),
    )
    finalizer(close_log!, engine)
    return engine
end

"""
    stop!(engine)

Signal the engine to stop its ZMQ listener loop.

Call this from another task or thread to gracefully shut down `start!`.
"""
function stop!(engine::ExecutionEngine)
    # Threads.Atomic supports getindex/setindex! (engine.should_stop[]);
    # Threads.atomic_store!/atomic_load are for AtomicMemory / lower-level APIs.
    engine.should_stop[] = true
    flush_log!(engine)
end

"""
    events(engine) -> Vector{SignalEvent}

Return the event log for this engine.
"""
events(engine::ExecutionEngine) = engine.events

"""
    portfolio_risk(engine) -> Float64

Current gross notional portfolio exposure: the sum over tickers of the absolute
position quantity multiplied by the last execution price seen for that ticker.
Positions are marked at their last trade price, not the live market.
"""
function portfolio_risk(engine::ExecutionEngine)
    total = 0.0
    for (ticker, quantity) in engine.positions
        price = get(engine.last_prices, ticker, 0.0)
        total += abs(quantity) * price
    end
    return total
end

# True when an order of `units` on `side` only reduces the existing signed position
# `current` (it neither increases |position| nor flips its side).
function _reduces_position(current::Float64, units::Float64, side::TradeSide)
    side == Buy && return current < 0.0 && units <= -current
    side == Sell && return current > 0.0 && units <= current
    return false
end

# Return `nothing` when the order passes both position caps, else an explicit reason
# naming the cap that was hit. Nothing is clamped: the caller rejects the whole order.
function _position_cap_violation(
    engine::ExecutionEngine,
    units::Float64,
    price::Float64,
    account_balance::Float64,
    current::Float64,
    side::TradeSide,
)
    if units > engine.max_position_size
        return "exceeds_max_position: units $(units) exceed max_position_size $(engine.max_position_size)"
    end
    units > 0.0 || return nothing
    _reduces_position(current, units, side) && return nothing
    notional = units * price
    limit = engine.max_position_fraction * account_balance
    # `!(notional <= limit)` also rejects a NaN notional or limit instead of passing it.
    if !(notional <= limit)
        return "exceeds_max_position_fraction: notional $(notional) (units $(units) × price $(price)) " *
               "exceeds max_position_fraction $(engine.max_position_fraction) × " *
               "account_balance $(account_balance) = $(limit)"
    end
    return nothing
end

"""
    execute_signal!(engine, signal, account_balance; observed_ns=nothing,
                    position_units_override=nothing) -> ExecutionDecision

Process one signal: gate by confidence, size via Kelly, update positions.

`observed_ns` supplies a deterministic observation timestamp for replay. Live callers
omit it to use the wall clock. `position_units_override` is reserved for simulators
that must apply an explicit filled quantity, such as a full-position close; the
decision event and position ledger both use the overridden quantity.

# Rejections
Every rejection increments `rejected_signals`, records a `SignalEvent`, and returns an
`ExecutionDecision` with `executed = false` and an explicit `reason`:

- `gate_reject`: confidence below the engine threshold.
- `neutral_reject`: `Neutral` side.
- `sizing_reject` (reason prefix `invalid_sizing_input:`): `size_position` rejected
  its inputs, e.g. `signal.price <= 0`, `account_balance <= 0`, a non-finite value, or
  confidence outside `[0, 1]`.
- `max_position_reject`: a position cap was hit. The reason names the cap:
  - prefix `exceeds_max_position:` when Kelly-sized units exceed the absolute
    `engine.max_position_size` (default `Inf`, no absolute cap);
  - prefix `exceeds_max_position_fraction:` when the order notional
    `units * price` exceeds `engine.max_position_fraction * account_balance`
    (default fraction `1.0`). Orders that only reduce the existing position in the
    ticker are exempt from this relative cap.
  When both caps are set, both apply and the absolute cap is checked first. Nothing is
  shrunk to a cap: the whole order is rejected and no position changes.
- `zero_reject`: the Kelly fraction is zero (no edge), so the sized position is zero.
- `portfolio_reject`: the projected gross exposure would exceed `max_portfolio_exposure`.

An invalid `position_units_override` (non-finite, negative, above `max_position_size`,
or, for an order that does not only reduce the position, with notional above
`max_position_fraction * account_balance`) is a caller error and throws
`ArgumentError` whose message carries the same cap reason. When an override
is supplied and `account_balance` is finite but `<= 0`, Kelly sizing is skipped (the
override quantity does not depend on it) so an existing position can still be closed;
price and confidence are still validated (as in `size_position`) and the reported
`kelly_fraction` is `0.0`. The carve-out
covers a finite balance `<= 0` only: a non-finite balance (`NaN`, `±Inf`) means the
accounting is already broken and is rejected as `sizing_reject` even with an override.

A valid input that yields zero Kelly units, such as no edge or `kelly_scalar = 0`, is a
real sizing decision and is reported as `zero_reject`, not as `sizing_reject`.

Behavior change: earlier versions silently capped Kelly-sized units with
`min(units, max_position_size)` and executed the reduced size, and treated
`account_balance <= 0` or `price <= 0` as a `zero_reject` with no specific reason.
The default cap is now relative (`max_position_fraction = 1.0`) instead of a fixed
`max_position_size = 10.0` units, which rejected ordinary Kelly-sized entries at a
`10_000` balance and prices below about `1_000 × k`.
"""
function execute_signal!(
    engine::ExecutionEngine,
    signal::TradeSignal,
    account_balance::Float64 = 10_000.0,
    ;
    observed_ns::Union{Nothing, Integer} = nothing,
    position_units_override::Union{Nothing, Real} = nothing,
)::ExecutionDecision
    engine.total_signals += 1
    lat = observed_ns === nothing ? latency_ns(signal) : latency_ns(signal, observed_ns)

    if !passes_gate(signal, engine.confidence_threshold)
        engine.rejected_signals += 1
        reason = "confidence=$(signal.confidence) < threshold=$(engine.confidence_threshold)"
        _record_event!(
            engine,
            SignalEvent(
                "gate_reject",
                signal.ticker,
                signal.confidence,
                string(signal.side);
                reason = reason,
                latency_ns = lat,
            ),
        )
        return ExecutionDecision(signal, false, reason, 0.0, 0.0, 0.0, lat)
    end

    if signal.side == Neutral
        engine.rejected_signals += 1
        _record_event!(
            engine,
            SignalEvent(
                "neutral_reject",
                signal.ticker,
                signal.confidence,
                string(signal.side);
                latency_ns = lat,
            ),
        )
        return ExecutionDecision(signal, false, "neutral signal", 0.0, 0.0, 0.0, lat)
    end

    # Invalid sizing inputs (non-positive or non-finite price or balance, confidence
    # outside [0, 1], invalid payoff ratio) are rejected with an explicit reason rather
    # than sized to zero or coerced into range.
    #
    # Exception: an explicit-fill override (e.g. a simulator closing an open position)
    # does not use the Kelly size, so it may proceed when the account balance is
    # finite but non-positive. Price and confidence are still validated, and the reported Kelly
    # fraction is 0.0 because no Kelly size was computed.
    sizing_error = nothing
    kelly = 0.0
    kelly_units = 0.0
    if position_units_override !== nothing && isfinite(account_balance) && account_balance <= 0.0
        # Same price and confidence validation as `size_position`; only the balance check
        # (which Kelly sizing needs and the override does not) is skipped.
        conf = Float64(signal.confidence)
        if !(isfinite(signal.price) && signal.price > 0.0)
            sizing_error = "price must be finite and positive, got $(repr(signal.price))"
        elseif !(isfinite(conf) && 0.0 <= conf <= 1.0)
            sizing_error = "confidence must be a finite probability in [0, 1], got $(repr(conf))"
        end
    else
        try
            position = size_position(
                confidence = Float64(signal.confidence),
                price = signal.price,
                account_balance = account_balance,
                payoff_ratio = engine.payoff_ratio,
            )
            kelly = position.kelly_fraction
            kelly_units = position.units
        catch e
            e isa ArgumentError || rethrow()
            sizing_error = e.msg
        end
    end
    if sizing_error !== nothing
        engine.rejected_signals += 1
        reason = "invalid_sizing_input: $(sizing_error)"
        _record_event!(
            engine,
            SignalEvent(
                "sizing_reject",
                signal.ticker,
                signal.confidence,
                string(signal.side);
                reason = reason,
                latency_ns = lat,
            ),
        )
        return ExecutionDecision(signal, false, reason, 0.0, 0.0, 0.0, lat)
    end

    price = Float64(signal.price)
    current = get(engine.positions, signal.ticker, 0.0)
    units = if position_units_override === nothing
        kelly_units
    else
        overridden = Float64(position_units_override)
        if !isfinite(overridden) || overridden < 0.0
            throw(ArgumentError("position_units_override must be finite and non-negative"))
        end
        overridden
    end

    # Position caps reject, never shrink (research invariant 7). The absolute unit cap
    # applies to every order. The relative notional cap applies to every order except one
    # that only reduces the existing position in this ticker: price and balance drift
    # after entry, and a cap that blocks a reduction would trap the position.
    cap_reason =
        _position_cap_violation(engine, units, price, account_balance, current, signal.side)
    if cap_reason !== nothing
        if position_units_override !== nothing
            # An explicit fill quantity above a cap is a caller error, not a decision.
            throw(ArgumentError("position_units_override rejected: $(cap_reason)"))
        end
        engine.rejected_signals += 1
        _record_event!(
            engine,
            SignalEvent(
                "max_position_reject",
                signal.ticker,
                signal.confidence,
                string(signal.side);
                reason = cap_reason,
                kelly_fraction = kelly,
                latency_ns = lat,
            ),
        )
        return ExecutionDecision(signal, false, cap_reason, kelly, 0.0, 0.0, lat)
    end

    if units <= 0.0
        engine.rejected_signals += 1
        _record_event!(
            engine,
            SignalEvent(
                "zero_reject",
                signal.ticker,
                signal.confidence,
                string(signal.side);
                reason = "zero-sized position",
                kelly_fraction = kelly,
                latency_ns = lat,
            ),
        )
        return ExecutionDecision(signal, false, "zero-sized position", kelly, 0.0, 0.0, lat)
    end

    applied_fraction = account_balance <= 0.0 ? 0.0 : (units * signal.price) / account_balance

    # Portfolio exposure cap: gross notional (Σ |quantity| × last trade price)
    # after applying this signal must not exceed the cap. Signals that only
    # reduce the existing position in the same ticker lower exposure and are
    # always allowed.
    projected_quantity = signal.side == Buy ? current + units : current - units
    prior_ticker_exposure = abs(current) * get(engine.last_prices, signal.ticker, 0.0)
    projected_ticker_exposure = abs(projected_quantity) * price
    projected_exposure = portfolio_risk(engine) - prior_ticker_exposure + projected_ticker_exposure
    if projected_exposure > engine.max_portfolio_exposure
        engine.rejected_signals += 1
        reason = "portfolio exposure $(projected_exposure) would exceed cap $(engine.max_portfolio_exposure)"
        _record_event!(
            engine,
            SignalEvent(
                "portfolio_reject",
                signal.ticker,
                signal.confidence,
                string(signal.side);
                reason = reason,
                kelly_fraction = kelly,
                latency_ns = lat,
            ),
        )
        return ExecutionDecision(signal, false, reason, kelly, 0.0, 0.0, lat)
    end

    # Update position book
    if signal.side == Buy
        engine.positions[signal.ticker] = current + units
    elseif signal.side == Sell
        engine.positions[signal.ticker] = current - units
    end
    engine.last_prices[signal.ticker] = price

    engine.executed_signals += 1
    _record_event!(
        engine,
        SignalEvent(
            "executed",
            signal.ticker,
            signal.confidence,
            string(signal.side);
            kelly_fraction = kelly,
            latency_ns = lat,
            executed = true,
            position_units = units,
            applied_fraction = applied_fraction,
        ),
    )
    return ExecutionDecision(signal, true, "executed", kelly, applied_fraction, units, lat)
end

"""
    fill_rate(engine) -> Float64

Fraction of signals that were executed (not rejected).
"""
fill_rate(e::ExecutionEngine) = e.total_signals == 0 ? 0.0 : e.executed_signals / e.total_signals

"""
    load_history(path) -> Vector{SignalEvent}

Load a JSON-lines event history written by an `ExecutionEngine` with `log_file`.
Returns an empty vector for missing or empty files, and skips malformed lines
with a warning.
"""
function load_history(path::String)::Vector{SignalEvent}
    events = SignalEvent[]
    isfile(path) || return events

    open(path, "r") do io
        for (i, line) in enumerate(eachline(io))
            s = strip(line)
            isempty(s) && continue
            try
                d = JSON.parse(s)
                if get(d, "schema_version", _EVENT_SCHEMA_VERSION) != _EVENT_SCHEMA_VERSION
                    @warn "Skipping event log line $i with unexpected schema version"
                    continue
                end
                push!(events, SignalEvent(d))
            catch e
                @warn "Skipping malformed event log line $i: $e"
            end
        end
    end

    return events
end

# ── dYdX v4 REST Client ───────────────────────────────────────────────────────

"""
    DydxPrice

Current price data from dYdX v4 indexer.
"""
struct DydxPrice
    ticker::String
    oracle_price::Float64
    best_bid::Float64
    best_ask::Float64
end

"""
    mid_price(p) -> Float64

Mid-price between best bid and ask.
"""
mid_price(p::DydxPrice) = (p.best_bid + p.best_ask) / 2.0

"""
    spread_bps(p) -> Float64

Bid-ask spread in basis points.
"""
spread_bps(p::DydxPrice) = max(p.best_ask - p.best_bid, 0.0) / max(p.best_ask, 1e-9) * 10_000.0

# ── Rate Limiter ──────────────────────────────────────────────────────────────

"""
    RateLimiter

Token bucket rate limiter for API calls.

# Fields
- `tokens`:        current available tokens
- `max_tokens`:    maximum token capacity
- `refill_rate`:   tokens added per second
- `last_refill`:   timestamp of last refill
- `lock`:          thread-safe lock
"""
mutable struct RateLimiter
    tokens::Float64
    max_tokens::Float64
    refill_rate::Float64
    last_refill::Float64
    lock::ReentrantLock
end

"""
    RateLimiter(; requests_per_second=10.0, burst=10.0)

Create a rate limiter with the given requests per second and burst capacity.
"""
function RateLimiter(; requests_per_second::Float64 = 10.0, burst::Float64 = 10.0)
    if requests_per_second <= 0
        throw(ArgumentError("requests_per_second must be > 0, got: $requests_per_second"))
    end
    if burst <= 0
        throw(ArgumentError("burst must be > 0, got: $burst"))
    end
    RateLimiter(burst, burst, requests_per_second, time(), ReentrantLock())
end

"""
    acquire!(limiter)

Block until a token is available. Refills tokens based on elapsed time.
Thread-safe via ReentrantLock.
"""
function acquire!(limiter::RateLimiter)
    lock(limiter.lock) do
        if limiter.refill_rate <= 0
            throw(ArgumentError("refill_rate must be > 0, got: $(limiter.refill_rate)"))
        end

        now = time()
        elapsed = now - limiter.last_refill
        limiter.tokens = min(limiter.max_tokens, limiter.tokens + elapsed * limiter.refill_rate)
        limiter.last_refill = now

        if limiter.tokens < 1.0
            wait_time = (1.0 - limiter.tokens) / limiter.refill_rate
            sleep(wait_time)

            # Refill again after sleep (preserve any fractional credit)
            now2 = time()
            elapsed2 = now2 - limiter.last_refill
            limiter.tokens =
                min(limiter.max_tokens, limiter.tokens + elapsed2 * limiter.refill_rate)
            limiter.last_refill = now2
        end

        limiter.tokens = max(0.0, limiter.tokens - 1.0)
    end
end

"""
    set_rate!(limiter, requests_per_second)

Update the rate limit at runtime.
"""
function set_rate!(limiter::RateLimiter, requests_per_second::Float64)
    if requests_per_second <= 0
        throw(ArgumentError("requests_per_second must be > 0, got: $requests_per_second"))
    end

    lock(limiter.lock) do
        # Refill tokens based on current state before changing rate
        now = time()
        elapsed = now - limiter.last_refill
        limiter.tokens = min(limiter.max_tokens, limiter.tokens + elapsed * limiter.refill_rate)
        limiter.last_refill = now

        # Preserve burst capacity (max_tokens) and only change refill behavior.
        limiter.refill_rate = requests_per_second
    end
end

# ── Price Cache ───────────────────────────────────────────────────────────────

"""
    PriceCache

Short TTL cache for dYdX price data.

# Fields
- `prices`:   cached DydxPrice per ticker
- `times`:    timestamp of last fetch per ticker
- `ttl_s`:    time-to-live in seconds
- `lock`:     thread-safe lock
"""
mutable struct PriceCache
    prices::Dict{String, DydxPrice}
    times::Dict{String, Float64}
    ttl_s::Float64
    lock::ReentrantLock
end

"""
    PriceCache(; ttl_s=5.0)

Create a price cache with the given TTL in seconds.
"""
function PriceCache(; ttl_s::Float64 = 5.0)
    PriceCache(Dict{String, DydxPrice}(), Dict{String, Float64}(), ttl_s, ReentrantLock())
end

"""
    invalidate!(cache, ticker)

Remove a specific ticker from the cache.
"""
function invalidate!(cache::PriceCache, ticker::String)
    lock(cache.lock) do
        delete!(cache.prices, ticker)
        delete!(cache.times, ticker)
    end
end

"""
    clear!(cache)

Remove all entries from the cache.
"""
function clear!(cache::PriceCache)
    lock(cache.lock) do
        empty!(cache.prices)
        empty!(cache.times)
    end
end

"""
    cache_size(cache)

Return the number of entries in the cache.
"""
function cache_size(cache::PriceCache)
    lock(cache.lock) do
        return length(cache.prices)
    end
end

"""
    is_fresh(cache, ticker)

Check if a cached entry is still within TTL.
"""
function is_fresh(cache::PriceCache, ticker::String)
    lock(cache.lock) do
        if !haskey(cache.times, ticker)
            return false
        end
        return (time() - cache.times[ticker]) < cache.ttl_s
    end
end

"""
    get_cached(cache, ticker)

Get a cached price if fresh, nothing otherwise.
"""
function get_cached(cache::PriceCache, ticker::String)
    lock(cache.lock) do
        if haskey(cache.prices, ticker) && haskey(cache.times, ticker)
            if (time() - cache.times[ticker]) < cache.ttl_s
                return cache.prices[ticker]
            end
            # Lazy eviction: drop stale entries so cache_size stays accurate
            delete!(cache.prices, ticker)
            delete!(cache.times, ticker)
        end
        return nothing
    end
end

"""
    put_cached!(cache, ticker, price)

Store a price in the cache with current timestamp.
"""
function put_cached!(cache::PriceCache, ticker::String, price::DydxPrice)
    lock(cache.lock) do
        cache.prices[ticker] = price
        cache.times[ticker] = time()
    end
end

# ── dYdX v4 REST Client ───────────────────────────────────────────────────────

"""
    DydxClient

REST client for dYdX v4 perpetuals (no API key required for read-only market data).
"""
struct DydxClient
    base_url::String
    timeout_s::Float64
    rate_limiter::Union{RateLimiter, Nothing}
    cache::Union{PriceCache, Nothing}
end

"""
    DydxClient(; base_url, timeout_s, rate_limit, cache)

Create a dYdX client. Pass `rate_limit=RateLimiter()` to enable rate limiting.
Pass `cache=PriceCache()` to enable price caching.
"""
function DydxClient(;
    base_url::String = "https://indexer.dydx.trade/v4",
    timeout_s::Float64 = 5.0,
    rate_limit::Union{RateLimiter, Nothing} = nothing,
    cache::Union{PriceCache, Nothing} = nothing,
)
    DydxClient(base_url, timeout_s, rate_limit, cache)
end

"""
    get_price(client, ticker) -> Union{DydxPrice, Nothing}

Fetch current oracle price and order book top for `ticker`.
Returns `nothing` on network error. Uses cache if configured.
"""
function get_price(client::DydxClient, ticker::String)::Union{DydxPrice, Nothing}
    # Check cache first
    if client.cache !== nothing
        cached = get_cached(client.cache, ticker)
        if cached !== nothing
            return cached
        end
    end

    try
        # Rate limit if configured
        if client.rate_limiter !== nothing
            acquire!(client.rate_limiter)
        end

        url = "$(client.base_url)/orderbooks/perpetualMarket/$(ticker)"
        resp = HTTP.get(url; readtimeout = client.timeout_s)
        data = JSON.parse(String(resp.body))

        bids = get(data, "bids", [])
        asks = get(data, "asks", [])

        best_bid = isempty(bids) ? 0.0 : parse(Float64, first(bids)["price"])
        best_ask = isempty(asks) ? 0.0 : parse(Float64, first(asks)["price"])

        # Rate limit for second call
        if client.rate_limiter !== nothing
            acquire!(client.rate_limiter)
        end

        # Oracle price from markets endpoint
        market_url = "$(client.base_url)/perpetualMarkets?ticker=$(ticker)"
        market_resp = HTTP.get(market_url; readtimeout = client.timeout_s)
        market_data = JSON.parse(String(market_resp.body))
        markets = get(market_data, "markets", Dict())
        oracle = if haskey(markets, ticker)
            parse(Float64, get(markets[ticker], "oraclePrice", "0"))
        else
            (best_bid + best_ask) / 2.0
        end

        price = DydxPrice(ticker, oracle, best_bid, best_ask)

        # Store in cache
        if client.cache !== nothing
            put_cached!(client.cache, ticker, price)
        end

        return price
    catch e
        e isa InterruptException && rethrow()
        @warn "dYdX price fetch failed for $ticker: $e"
        return nothing
    end
end

# ── ZMQ Signal Listener ───────────────────────────────────────────────────────

"""
    strip_zmq_topic(msg, topic) -> String

Remove a ZMQ topic prefix from `msg` using byte (codeunit) length so multi-byte
UTF-8 topics are stripped correctly. Falls back to the full message when the
topic is empty or not a prefix.
"""
function strip_zmq_topic(msg::AbstractString, topic::AbstractString)
    if isempty(topic) || !startswith(msg, topic)
        return String(msg)
    end
    # ncodeunits is the correct end-of-prefix byte index for UTF-8 strings
    prefix_bytes = ncodeunits(topic)
    if prefix_bytes >= ncodeunits(msg)
        return ""
    end
    return msg[nextind(msg, prefix_bytes):end]
end

"""
    start!(engine; zmq_endpoint, zmq_topic, account_balance, on_decision,
           timeout_s, reconnect_interval_s, max_reconnect_attempts)

Start the ZMQ SUB listener loop (requires ZMQ.jl).

Subscribes to `zmq_endpoint` and processes incoming JSON trade signals
through the execution engine, calling `on_decision` for each result.

# Arguments
- `engine`:                   `ExecutionEngine` instance
- `zmq_endpoint`:             ZMQ endpoint (e.g. "tcp://localhost:5555" or "ipc:///tmp/signals.ipc")
- `zmq_topic`:                ZMQ topic filter prefix (default "" = all messages)
- `account_balance`:          account size for Kelly sizing
- `on_decision`:              callback `(ExecutionDecision) -> Nothing`
- `timeout_s`:                total seconds to run before auto-stopping (nothing = run forever)
- `reconnect_interval_s`:     seconds to wait between reconnection attempts (default 5.0)
- `max_reconnect_attempts`:   max reconnection attempts; 0 = infinite (default 0)

# Shutdown

Call `stop!(engine)` from another task to gracefully stop the listener.
The loop checks `engine.should_stop` periodically.

# Example
```julia
engine = ExecutionEngine(confidence_threshold=Float32(0.85))

# Run in background task
t = @async start!(
    engine,
    zmq_endpoint="tcp://localhost:5555",
    zmq_topic="trade.signal",
    on_decision = decision -> println("Decision: \$(decision.executed)"),
)

# Stop after 60 seconds
sleep(60)
stop!(engine)
```
"""
function start!(
    engine::ExecutionEngine;
    zmq_endpoint::String = "tcp://localhost:5555",
    zmq_topic::String = "",
    account_balance::Float64 = 10_000.0,
    on_decision = decision -> nothing,
    timeout_s::Union{Float64, Nothing} = nothing,
    reconnect_interval_s::Float64 = 5.0,
    max_reconnect_attempts::Int = 0,
)
    # Allow engine reuse across stop/start cycles.
    engine.should_stop[] = false

    start_time = time()
    reconnect_count = 0

    while !engine.should_stop[]
        if timeout_s !== nothing && (time() - start_time) >= timeout_s
            @info "[execution] Timeout reached, stopping listener"
            break
        end

        ctx = nothing
        socket = nothing
        received_ok = false
        try
            # Construct inside try so allocation failures are reconnectable
            # and do not leak a Context if Socket construction throws.
            ctx = ZMQ.Context()
            socket = ZMQ.Socket(ctx, ZMQ.SUB)
            ZMQ.subscribe(socket, zmq_topic)
            ZMQ.connect(socket, zmq_endpoint)

            # Set receive timeout to 1 second so we can check should_stop periodically
            socket.rcvtimeo = 1000

            @info "[execution] ZMQ SUB connected to $zmq_endpoint (topic=\"$(zmq_topic)\")"
            # NOTE: ZMQ.connect is asynchronous and succeeds even with no live
            # peer, so do NOT treat it as healthy progress here. Only reset the
            # reconnect budget after actually receiving/processing a message
            # (see received_ok = true in the recv path below).

            while !engine.should_stop[]
                if timeout_s !== nothing && (time() - start_time) >= timeout_s
                    @info "[execution] Timeout reached, stopping listener"
                    break
                end

                try
                    msg = String(ZMQ.recv(socket))
                    payload = strip_zmq_topic(msg, zmq_topic)
                    data = JSON.parse(payload)

                    # Validate signal before processing
                    err = validate_signal(data)
                    if err !== nothing
                        @warn "[execution] Invalid signal: $err"
                        continue
                    end

                    signal = TradeSignal(data)
                    decision = execute_signal!(engine, signal, account_balance)
                    on_decision(decision)
                    # Also mark healthy on successful message processing
                    received_ok = true
                catch e
                    if e isa ZMQ.TimeoutError
                        # recv timeout, loop back to check should_stop
                        continue
                    elseif e isa ZMQ.StateError
                        @warn "[execution] ZMQ transport error, reconnecting: $e"
                        break  # break inner loop to reconnect
                    else
                        # Application-level errors (JSON, callback): log, keep listening
                        @warn "[execution] Message handling error (not reconnecting): $e"
                        continue
                    end
                end
            end
        catch e
            if e isa InterruptException
                rethrow()
            else
                @warn "[execution] ZMQ connection error: $e"
            end
        finally
            if socket !== nothing
                try
                    ZMQ.close(socket)
                catch
                end
            end
            if ctx !== nothing
                try
                    ZMQ.close(ctx)
                catch
                end
            end
        end

        # Reset reconnect counter after successful message processing
        if received_ok
            reconnect_count = 0
        end

        # If we exited normally (should_stop or timeout), don't reconnect
        if engine.should_stop[]
            break
        end
        if timeout_s !== nothing && (time() - start_time) >= timeout_s
            break
        end

        # Reconnect logic
        reconnect_count += 1
        if max_reconnect_attempts > 0 && reconnect_count >= max_reconnect_attempts
            @warn "[execution] Max reconnect attempts ($max_reconnect_attempts) reached, stopping"
            break
        end

        @info "[execution] Reconnecting in $(reconnect_interval_s)s (attempt $reconnect_count)..."
        # Sleep in small chunks so stop!() or timeout can take effect promptly
        elapsed = 0.0
        while elapsed < reconnect_interval_s && !engine.should_stop[]
            if timeout_s !== nothing && (time() - start_time) >= timeout_s
                break
            end
            sleep(min(0.5, reconnect_interval_s - elapsed))
            elapsed += 0.5
        end
    end

    flush_log!(engine)
    @info "[execution] ZMQ listener stopped"
end

# ── Backtest Harness ──────────────────────────────────────────────────────────

include("backtest/Backtest.jl")
using .Backtest

end # module
