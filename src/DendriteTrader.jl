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
using ZMQ

export TradeSignal,
    TradeSide,
    Buy,
    Sell,
    Neutral,
    ExecutionEngine,
    ExecutionDecision,
    SignalEvent,
    load_history,
    close_log!
export validate_signal, execute_signal!, latency_ns, passes_gate
export DydxClient, DydxPrice, get_price, mid_price, spread_bps
export RateLimiter, acquire!, set_rate!
export PriceCache, invalidate!, clear!, cache_size, get_cached, put_cached!, is_fresh
export start!, stop!, events, fill_rate
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
    validate_signal(d::Dict) -> Union{Nothing, String}

Validate a trade signal dictionary. Returns `nothing` if valid, or an error message string.

# Required Fields
- `ticker`: String, non-empty
- `side`: "BUY", "SELL", or "NEUTRAL"
- `price`: Number > 0
- `confidence`: Number in [0.0, 1.0]
- `timestamp_ns`: Integer > 0
"""
function validate_signal(d::Dict)
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

function TradeSignal(d::Dict)
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
    SignalEvent(d::Dict)

Deserialize a SignalEvent from a dictionary (typically parsed from a JSON line).
"""
function SignalEvent(d::Dict)
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

"""
    ExecutionEngine

Stateful execution engine with confidence gating and position management.

# Fields
- `confidence_threshold`: minimum SNN confidence to execute (default 0.85)
- `max_position_size`:    hard cap on position units (default 10.0)
- `payoff_ratio`:         odds-style average win/loss ratio for Kelly sizing (default 1.5)
- `positions`:            current open positions (ticker → quantity)
- `log_file`:             optional path to a JSON-lines event log
- `log_io`:               open file handle for the event log (if configured)
- `log_lock`:             lock serializing `log_io` writes
"""
mutable struct ExecutionEngine
    confidence_threshold::Float32
    max_position_size::Float64
    payoff_ratio::Float64
    positions::Dict{String, Float64}
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
    ExecutionEngine(; confidence_threshold=0.85f0, max_position_size=10.0, payoff_ratio=1.5, log_file=nothing, truncate=false)

Create an execution engine. If `log_file` is provided, events are appended as JSON-lines.
Pass `truncate=true` to truncate an existing log.
"""
function ExecutionEngine(;
    confidence_threshold::Float32 = Float32(0.85),
    max_position_size::Float64 = 10.0,
    payoff_ratio::Float64 = 1.5,
    log_file::Union{String, Nothing} = nothing,
    truncate::Bool = false,
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
        payoff_ratio,
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
    execute_signal!(engine, signal, account_balance) -> ExecutionDecision

Process one signal: gate by confidence, size via Kelly, update positions.
"""
function execute_signal!(
    engine::ExecutionEngine,
    signal::TradeSignal,
    account_balance::Float64 = 10_000.0,
)::ExecutionDecision
    engine.total_signals += 1
    lat = latency_ns(signal)

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

    position = size_position(
        confidence = Float64(signal.confidence),
        price = signal.price,
        account_balance = account_balance,
        payoff_ratio = engine.payoff_ratio,
    )
    units = min(position.units, engine.max_position_size)

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
                kelly_fraction = position.kelly_fraction,
                latency_ns = lat,
            ),
        )
        return ExecutionDecision(
            signal,
            false,
            "zero-sized position",
            position.kelly_fraction,
            0.0,
            0.0,
            lat,
        )
    end

    applied_fraction = account_balance <= 0.0 ? 0.0 : (units * signal.price) / account_balance

    # Update position book
    current = get(engine.positions, signal.ticker, 0.0)
    if signal.side == Buy
        engine.positions[signal.ticker] = current + units
    elseif signal.side == Sell
        engine.positions[signal.ticker] = current - units
    end

    engine.executed_signals += 1
    _record_event!(
        engine,
        SignalEvent(
            "executed",
            signal.ticker,
            signal.confidence,
            string(signal.side);
            kelly_fraction = position.kelly_fraction,
            latency_ns = lat,
            executed = true,
            position_units = units,
            applied_fraction = applied_fraction,
        ),
    )
    return ExecutionDecision(
        signal,
        true,
        "executed",
        position.kelly_fraction,
        applied_fraction,
        units,
        lat,
    )
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
