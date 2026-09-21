# SPDX-License-Identifier: MIT OR Apache-2.0

struct ReplayPolicy
    allow_sequence_gaps::Bool
end

ReplayPolicy(; allow_sequence_gaps::Bool = false) = ReplayPolicy(allow_sequence_gaps)

struct ReplaySession <: AbstractVector{MarketEvent}
    venue::Symbol
    instrument::String
    events::Vector{MarketEvent}
    policy::ReplayPolicy

    function ReplaySession(
        events::AbstractVector{<:MarketEvent};
        policy::ReplayPolicy = ReplayPolicy(),
    )
        isempty(events) && throw(ArgumentError("replay session must contain events"))
        venue = first(events).venue
        instrument = first(events).instrument
        all(event -> event.venue == venue, events) ||
            throw(ArgumentError("replay session contains multiple venues"))
        all(event -> event.instrument == instrument, events) ||
            throw(ArgumentError("replay session contains multiple instruments"))
        return new(venue, instrument, MarketEvent[events...], policy)
    end
end

Base.IndexStyle(::Type{ReplaySession}) = IndexLinear()
Base.size(session::ReplaySession) = size(session.events)
Base.getindex(session::ReplaySession, index::Int) = session.events[index]

mutable struct OrderBookState
    venue::Symbol
    instrument::String
    bids::Dict{Int64, Int64}
    asks::Dict{Int64, Int64}
    exchange_ts_ns::Int64
    receive_ts_ns::Int64
    sequence::Int64
end

function OrderBookState(venue::Symbol, instrument::AbstractString)
    _validate_identity(venue, instrument)
    OrderBookState(
        venue,
        String(instrument),
        Dict{Int64, Int64}(),
        Dict{Int64, Int64}(),
        Int64(0),
        Int64(0),
        Int64(0),
    )
end

function _validate_order(book::OrderBookState, event::MarketEvent, policy::ReplayPolicy)
    event.venue == book.venue || throw(ArgumentError("event venue does not match book"))
    event.instrument == book.instrument ||
        throw(ArgumentError("event instrument does not match book"))
    event.sequence > book.sequence || throw(ArgumentError("sequence must increase"))
    event.exchange_ts_ns >= book.exchange_ts_ns ||
        throw(ArgumentError("exchange timestamp must not decrease"))
    event.receive_ts_ns >= book.receive_ts_ns ||
        throw(ArgumentError("receive timestamp must not decrease"))
    if book.sequence != 0 && !policy.allow_sequence_gaps
        event.sequence == book.sequence + 1 || throw(ArgumentError("sequence gap"))
    end
    return nothing
end

function _advance!(book::OrderBookState, event::MarketEvent)
    book.exchange_ts_ns = event.exchange_ts_ns
    book.receive_ts_ns = event.receive_ts_ns
    book.sequence = event.sequence
    return book
end

function apply!(book::OrderBookState, event::BookDelta; policy::ReplayPolicy = ReplayPolicy())
    _validate_order(book, event, policy)
    levels = event.side == Bid ? book.bids : book.asks
    new_size = get(levels, event.price_ticks, Int64(0)) + event.size_delta
    new_size >= 0 || throw(ArgumentError("book depth cannot become negative"))

    if new_size > 0 && event.side == Bid && !isempty(book.asks)
        event.price_ticks <= minimum(keys(book.asks)) ||
            throw(ArgumentError("bid update would cross the book"))
    elseif new_size > 0 && event.side == Ask && !isempty(book.bids)
        event.price_ticks >= maximum(keys(book.bids)) ||
            throw(ArgumentError("ask update would cross the book"))
    end

    if new_size == 0
        delete!(levels, event.price_ticks)
    else
        levels[event.price_ticks] = new_size
    end
    return _advance!(book, event)
end

function apply!(book::OrderBookState, event::TradePrint; policy::ReplayPolicy = ReplayPolicy())
    _validate_order(book, event, policy)
    return _advance!(book, event)
end

function snapshot(book::OrderBookState; depth::Integer = typemax(Int))
    depth >= 0 || throw(ArgumentError("depth must be non-negative"))
    bids = sort!(collect(book.bids); by = first, rev = true)
    asks = sort!(collect(book.asks); by = first)
    limit = Int(depth)
    resize!(bids, min(length(bids), limit))
    resize!(asks, min(length(asks), limit))
    return BookSnapshot(
        book.venue,
        book.instrument,
        book.exchange_ts_ns,
        book.receive_ts_ns,
        book.sequence,
        bids,
        asks,
    )
end

best_bid(book::OrderBookState) =
    isempty(book.bids) ? nothing : maximum(keys(book.bids)) => book.bids[maximum(keys(book.bids))]
best_ask(book::OrderBookState) =
    isempty(book.asks) ? nothing : minimum(keys(book.asks)) => book.asks[minimum(keys(book.asks))]

function spread_ticks(book::OrderBookState)
    bid = best_bid(book)
    ask = best_ask(book)
    return isnothing(bid) || isnothing(ask) ? nothing : first(ask) - first(bid)
end
