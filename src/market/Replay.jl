# SPDX-License-Identifier: MIT OR Apache-2.0

function _copy_book(book::OrderBookState)
    return OrderBookState(
        book.venue,
        book.instrument,
        copy(book.bids),
        copy(book.asks),
        book.exchange_ts_ns,
        book.receive_ts_ns,
        book.sequence,
    )
end

function _commit_book!(book::OrderBookState, source::OrderBookState)
    empty!(book.bids)
    merge!(book.bids, source.bids)
    empty!(book.asks)
    merge!(book.asks, source.asks)
    book.exchange_ts_ns = source.exchange_ts_ns
    book.receive_ts_ns = source.receive_ts_ns
    book.sequence = source.sequence
    return book
end

"""
    replay!(book, events; policy=ReplayPolicy(), depth=typemax(Int))

Transactionally apply a batch and return its snapshots. Any invalid argument or
event leaves `book` exactly as it was before the call.
"""
function replay!(
    book::OrderBookState,
    events::AbstractVector{<:MarketEvent};
    policy::ReplayPolicy = ReplayPolicy(),
    depth::Integer = typemax(Int),
)
    0 <= depth <= typemax(Int) ||
        throw(ArgumentError("depth must be between zero and typemax(Int)"))
    scratch = _copy_book(book)
    snapshots = BookSnapshot[]
    sizehint!(snapshots, length(events))
    for event in events
        apply!(scratch, event; policy = policy)
        push!(snapshots, snapshot(scratch; depth = depth))
    end
    _commit_book!(book, scratch)
    return snapshots
end

function replay!(book::OrderBookState, session::ReplaySession; depth::Integer = typemax(Int))
    book.venue == session.venue || throw(ArgumentError("session venue does not match book"))
    book.instrument == session.instrument ||
        throw(ArgumentError("session instrument does not match book"))
    return replay!(book, session.events; policy = session.policy, depth = depth)
end

function _required_integer(record::AbstractDict, field::String)
    value = get(record, field, nothing)
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$field must be an integer"))
    return value
end

function _required_string(record::AbstractDict, field::String)
    value = get(record, field, nothing)
    value isa AbstractString || throw(ArgumentError("$field must be a string"))
    isempty(value) && throw(ArgumentError("$field must be non-empty"))
    return value
end

function _book_side(value::AbstractString)
    value == "BID" && return Bid
    value == "ASK" && return Ask
    throw(ArgumentError("side must be BID or ASK"))
end

function _trade_side(value::AbstractString)
    value == "BUY" && return Buy
    value == "SELL" && return Sell
    throw(ArgumentError("aggressor_side must be BUY or SELL"))
end

function _decode_event(record::AbstractDict)
    event_type = _required_string(record, "type")
    venue = Symbol(_required_string(record, "venue"))
    instrument = _required_string(record, "instrument")
    exchange_ts_ns = _required_integer(record, "exchange_ts_ns")
    receive_ts_ns = _required_integer(record, "receive_ts_ns")
    sequence = _required_integer(record, "sequence")
    price_ticks = _required_integer(record, "price_ticks")

    if event_type == "book_delta"
        return BookDelta(
            venue,
            instrument,
            exchange_ts_ns,
            receive_ts_ns,
            sequence,
            _book_side(_required_string(record, "side")),
            price_ticks,
            _required_integer(record, "size_delta"),
        )
    elseif event_type == "trade_print"
        return TradePrint(
            venue,
            instrument,
            exchange_ts_ns,
            receive_ts_ns,
            sequence,
            _trade_side(_required_string(record, "aggressor_side")),
            price_ticks,
            _required_integer(record, "size"),
        )
    end
    throw(ArgumentError("unknown market event type: $event_type"))
end

"""Load a non-empty, single-instrument `ReplaySession` from newline-delimited JSON."""
function load_session_jsonl(path::AbstractString; policy::ReplayPolicy = ReplayPolicy())
    events = MarketEvent[]
    open(path, "r") do io
        for (line_number, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            try
                record = JSON.parse(line)
                record isa AbstractDict || throw(ArgumentError("event must be a JSON object"))
                push!(events, _decode_event(record))
            catch error
                error isa InterruptException && rethrow()
                throw(
                    ArgumentError(
                        "invalid event at line $line_number: $(sprint(showerror, error))",
                    ),
                )
            end
        end
    end
    return ReplaySession(events; policy = policy)
end
