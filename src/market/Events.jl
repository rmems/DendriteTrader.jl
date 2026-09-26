# SPDX-License-Identifier: MIT OR Apache-2.0

@enum BookSide begin
    Bid = 1
    Ask = 2
end

abstract type MarketEvent end

"""
    BookDelta(...)

An integer-tick change to one L2 price level. `size_delta` is signed; a resulting
level size of zero removes the level during replay.
"""
struct BookDelta <: MarketEvent
    venue::Symbol
    instrument::String
    exchange_ts_ns::Int64
    receive_ts_ns::Int64
    sequence::Int64
    side::BookSide
    price_ticks::Int64
    size_delta::Int64

    function BookDelta(
        venue::Symbol,
        instrument::AbstractString,
        exchange_ts_ns::Integer,
        receive_ts_ns::Integer,
        sequence::Integer,
        side::BookSide,
        price_ticks::Integer,
        size_delta::Integer,
    )
        _validate_identity(venue, instrument)
        _validate_clock(exchange_ts_ns, receive_ts_ns, sequence)
        price_ticks > 0 || throw(ArgumentError("price_ticks must be positive"))
        size_delta != 0 || throw(ArgumentError("size_delta must be non-zero"))
        return new(
            venue,
            String(instrument),
            Int64(exchange_ts_ns),
            Int64(receive_ts_ns),
            Int64(sequence),
            side,
            Int64(price_ticks),
            Int64(size_delta),
        )
    end
end

"""An observed integer-tick trade with a non-neutral aggressor side."""
struct TradePrint <: MarketEvent
    venue::Symbol
    instrument::String
    exchange_ts_ns::Int64
    receive_ts_ns::Int64
    sequence::Int64
    aggressor_side::TradeSide
    price_ticks::Int64
    size::Int64

    function TradePrint(
        venue::Symbol,
        instrument::AbstractString,
        exchange_ts_ns::Integer,
        receive_ts_ns::Integer,
        sequence::Integer,
        aggressor_side::TradeSide,
        price_ticks::Integer,
        size::Integer,
    )
        _validate_identity(venue, instrument)
        _validate_clock(exchange_ts_ns, receive_ts_ns, sequence)
        aggressor_side != Neutral || throw(ArgumentError("aggressor_side must be Buy or Sell"))
        price_ticks > 0 || throw(ArgumentError("price_ticks must be positive"))
        size > 0 || throw(ArgumentError("size must be positive"))
        return new(
            venue,
            String(instrument),
            Int64(exchange_ts_ns),
            Int64(receive_ts_ns),
            Int64(sequence),
            aggressor_side,
            Int64(price_ticks),
            Int64(size),
        )
    end
end

"""A deterministic, price-sorted copy of an order book after one market event."""
struct BookSnapshot
    venue::Symbol
    instrument::String
    exchange_ts_ns::Int64
    receive_ts_ns::Int64
    sequence::Int64
    bids::Vector{Pair{Int64, Int64}}
    asks::Vector{Pair{Int64, Int64}}
end

function _validate_identity(venue::Symbol, instrument::AbstractString)
    isempty(String(venue)) && throw(ArgumentError("venue must be non-empty"))
    isempty(instrument) && throw(ArgumentError("instrument must be non-empty"))
    return nothing
end

function _validate_clock(exchange_ts_ns::Integer, receive_ts_ns::Integer, sequence::Integer)
    exchange_ts_ns > 0 || throw(ArgumentError("exchange_ts_ns must be positive"))
    receive_ts_ns > 0 || throw(ArgumentError("receive_ts_ns must be positive"))
    receive_ts_ns >= exchange_ts_ns ||
        throw(ArgumentError("receive_ts_ns must not precede exchange_ts_ns"))
    sequence > 0 || throw(ArgumentError("sequence must be positive"))
    return nothing
end

Base.:(==)(left::BookSnapshot, right::BookSnapshot) =
    left.venue == right.venue &&
    left.instrument == right.instrument &&
    left.exchange_ts_ns == right.exchange_ts_ns &&
    left.receive_ts_ns == right.receive_ts_ns &&
    left.sequence == right.sequence &&
    left.bids == right.bids &&
    left.asks == right.asks
