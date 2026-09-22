# SPDX-License-Identifier: MIT OR Apache-2.0

const FEATURE_NAMES =
    ("spread_ticks", "mid_ticks", "microprice_ticks", "imbalance", "signed_order_flow")

"""One causally computed microstructure observation at a replay sequence."""
struct FeatureRow
    exchange_ts_ns::Int64
    sequence::Int64
    spread_ticks::Float64
    mid_ticks::Float64
    microprice_ticks::Float64
    imbalance::Float64
    signed_order_flow::Float64

    function FeatureRow(
        exchange_ts_ns::Integer,
        sequence::Integer,
        spread_ticks::Real,
        mid_ticks::Real,
        microprice_ticks::Real,
        imbalance::Real,
        signed_order_flow::Real,
    )
        exchange_ts_ns > 0 || throw(ArgumentError("exchange_ts_ns must be positive"))
        sequence > 0 || throw(ArgumentError("sequence must be positive"))
        values = (
            Float64(spread_ticks),
            Float64(mid_ticks),
            Float64(microprice_ticks),
            Float64(imbalance),
            Float64(signed_order_flow),
        )
        all(isfinite, values) || throw(ArgumentError("feature values must be finite"))
        return new(Int64(exchange_ts_ns), Int64(sequence), values...)
    end
end

function _validate_feature_rows(rows)
    for index in 2:length(rows)
        rows[index].sequence > rows[index - 1].sequence ||
            throw(ArgumentError("feature sequences must strictly increase"))
        rows[index].exchange_ts_ns >= rows[index - 1].exchange_ts_ns ||
            throw(ArgumentError("feature timestamps must not decrease"))
    end
    return nothing
end

"""An ordered collection of `FeatureRow` values with protected row ordering."""
mutable struct FeatureFrame <: AbstractVector{FeatureRow}
    rows::Tuple{Vararg{FeatureRow}}

    function FeatureFrame(rows::AbstractVector{FeatureRow})
        _validate_feature_rows(rows)
        return new(tuple(rows...))
    end
end

Base.IndexStyle(::Type{FeatureFrame}) = IndexLinear()
Base.size(frame::FeatureFrame) = (length(frame.rows),)
Base.getindex(frame::FeatureFrame, index::Int) = frame.rows[index]

function Base.setproperty!(frame::FeatureFrame, name::Symbol, value)
    name == :rows || throw(ArgumentError("FeatureFrame only exposes the rows property"))
    replacement = FeatureFrame(FeatureRow[value...])
    return setfield!(frame, :rows, replacement.rows)
end

function _best_levels(snapshot::BookSnapshot)
    isempty(snapshot.bids) && return nothing
    isempty(snapshot.asks) && return nothing
    bid = first(snapshot.bids)
    bid_prices = Set{Int64}((first(bid),))
    for level in @view snapshot.bids[2:end]
        price = first(level)
        price ∉ bid_prices || throw(ArgumentError("bid price levels must be unique"))
        push!(bid_prices, price)
        price > first(bid) && (bid = level)
    end

    ask = first(snapshot.asks)
    ask_prices = Set{Int64}((first(ask),))
    for level in @view snapshot.asks[2:end]
        price = first(level)
        price ∉ ask_prices || throw(ArgumentError("ask price levels must be unique"))
        push!(ask_prices, price)
        price < first(ask) && (ask = level)
    end
    return bid, ask
end

function _validated_top_of_book(snapshot::BookSnapshot)
    levels = _best_levels(snapshot)
    isnothing(levels) && return nothing
    bid, ask = levels
    bid_price, bid_size = first(bid), last(bid)
    ask_price, ask_size = first(ask), last(ask)
    bid_price > 0 && ask_price > 0 || throw(ArgumentError("top-level prices must be positive"))
    bid_size > 0 && ask_size > 0 || throw(ArgumentError("top-level sizes must be positive"))
    ask_price >= bid_price || throw(ArgumentError("snapshot book is crossed"))
    return bid, ask
end

_exact_ratio_to_float64(numerator::Int128, denominator::Int128) = Float64(numerator // denominator)

function _validate_snapshot_session(snapshots::AbstractVector{BookSnapshot})
    isempty(snapshots) && return nothing
    venue = first(snapshots).venue
    instrument = first(snapshots).instrument
    previous_sequence = Int64(0)
    previous_exchange_timestamp = Int64(0)
    previous_receive_timestamp = Int64(0)
    for snapshot in snapshots
        isempty(String(snapshot.venue)) && throw(ArgumentError("venue must be non-empty"))
        isempty(snapshot.instrument) && throw(ArgumentError("instrument must be non-empty"))
        snapshot.venue == venue || throw(ArgumentError("snapshots must belong to one venue"))
        snapshot.instrument == instrument ||
            throw(ArgumentError("snapshots must belong to one instrument"))
        snapshot.exchange_ts_ns > 0 ||
            throw(ArgumentError("snapshot exchange timestamps must be positive"))
        snapshot.receive_ts_ns > 0 ||
            throw(ArgumentError("snapshot receive timestamps must be positive"))
        snapshot.receive_ts_ns >= snapshot.exchange_ts_ns ||
            throw(ArgumentError("snapshot receive timestamps must not precede exchange timestamps"))
        snapshot.sequence > previous_sequence ||
            throw(ArgumentError("snapshot sequences must strictly increase"))
        snapshot.exchange_ts_ns >= previous_exchange_timestamp ||
            throw(ArgumentError("snapshot exchange timestamps must not decrease"))
        snapshot.receive_ts_ns >= previous_receive_timestamp ||
            throw(ArgumentError("snapshot receive timestamps must not decrease"))
        previous_sequence = snapshot.sequence
        previous_exchange_timestamp = snapshot.exchange_ts_ns
        previous_receive_timestamp = snapshot.receive_ts_ns
    end
    return nothing
end

function _signed_order_flow(previous::BookSnapshot, current::BookSnapshot)
    previous_levels = _validated_top_of_book(previous)
    current_levels = _validated_top_of_book(current)
    isnothing(previous_levels) && return 0.0
    isnothing(current_levels) && return 0.0
    return _signed_order_flow(previous_levels, current_levels)
end

function _signed_order_flow(
    previous_levels::Tuple{Pair{Int64, Int64}, Pair{Int64, Int64}},
    current_levels::Tuple{Pair{Int64, Int64}, Pair{Int64, Int64}},
)
    previous_bid, previous_ask = previous_levels
    current_bid, current_ask = current_levels
    bid_flow = if first(current_bid) > first(previous_bid)
        Int128(last(current_bid))
    elseif first(current_bid) == first(previous_bid)
        Int128(last(current_bid)) - Int128(last(previous_bid))
    else
        -Int128(last(previous_bid))
    end
    ask_flow = if first(current_ask) < first(previous_ask)
        Int128(last(current_ask))
    elseif first(current_ask) == first(previous_ask)
        Int128(last(current_ask)) - Int128(last(previous_ask))
    else
        -Int128(last(previous_ask))
    end
    return Float64(bid_flow - ask_flow)
end

"""
    microstructure_features(snapshots; allow_sequence_gaps=false) -> FeatureFrame

Compute spread, mid-price, microprice, top-level imbalance, and signed order flow
using only the current and immediately preceding complete snapshot. Snapshots lacking
either side are omitted. Sequence gaps are rejected by default because signed order
flow requires adjacent exchange events. Set `allow_sequence_gaps=true` only when
the input is intentionally an observed-event stream.
"""
function microstructure_features(
    snapshots::AbstractVector{BookSnapshot};
    allow_sequence_gaps::Bool = false,
)
    _validate_snapshot_session(snapshots)
    if !allow_sequence_gaps
        for index in 2:length(snapshots)
            snapshots[index].sequence == snapshots[index - 1].sequence + 1 ||
                throw(ArgumentError("signed order flow requires contiguous sequences"))
        end
    end
    rows = FeatureRow[]
    previous_complete_levels = nothing

    for current in snapshots
        levels = _validated_top_of_book(current)
        isnothing(levels) && continue
        bid, ask = levels
        bid_price, bid_size = first(bid), last(bid)
        ask_price, ask_size = first(ask), last(ask)
        bid_size_wide = Int128(bid_size)
        ask_size_wide = Int128(ask_size)
        total_size = bid_size_wide + ask_size_wide
        spread = Float64(Int128(ask_price) - Int128(bid_price))
        mid = _exact_ratio_to_float64(Int128(bid_price) + Int128(ask_price), Int128(2))
        microprice = _exact_ratio_to_float64(
            Int128(ask_price) * bid_size_wide + Int128(bid_price) * ask_size_wide,
            total_size,
        )
        imbalance = _exact_ratio_to_float64(bid_size_wide - ask_size_wide, total_size)
        order_flow =
            isnothing(previous_complete_levels) ? 0.0 :
            _signed_order_flow(previous_complete_levels, levels)
        push!(
            rows,
            FeatureRow(
                current.exchange_ts_ns,
                current.sequence,
                spread,
                mid,
                microprice,
                imbalance,
                order_flow,
            ),
        )
        previous_complete_levels = levels
    end
    return FeatureFrame(rows)
end
