# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Deterministic signed delta encoder. For each feature channel, the floored scaled delta
count (at least one once the threshold is reached) is emitted, capped at
`saturation`; the sign carries delta direction. Resetting drops the prior sample,
so the first encoded row is silent.
"""
mutable struct DeltaEncoder <: AbstractSpikeEncoder
    scale::Float64
    threshold::Float64
    saturation::Int8
    previous::Union{Nothing, Vector{Float64}}
    previous_exchange_ts_ns::Int64
    previous_sequence::Int64
    function DeltaEncoder(; scale::Real = 1.0, threshold::Real = 1.0, saturation::Integer = 1)
        scale_float, threshold_float = Float64(scale), Float64(threshold)
        isfinite(scale_float) && scale_float > 0 ||
            throw(ArgumentError("scale must be finite and positive"))
        isfinite(threshold_float) && threshold_float >= 0 ||
            throw(ArgumentError("threshold must be finite and non-negative"))
        1 <= saturation <= typemax(Int8) || throw(ArgumentError("saturation must be in 1:127"))
        new(scale_float, threshold_float, Int8(saturation), nothing, 0, 0)
    end
end

function reset_state!(encoder::DeltaEncoder)
    encoder.previous = nothing
    encoder.previous_exchange_ts_ns = 0
    encoder.previous_sequence = 0
    return encoder
end

function encode!(encoder::DeltaEncoder, frame::FeatureFrame)
    spikes = zeros(Int8, length(frame), 5)
    for (index, row) in enumerate(frame)
        current = _spike_features(row)
        if !isnothing(encoder.previous)
            row.sequence > encoder.previous_sequence || throw(
                ArgumentError("encoded feature sequences must strictly increase across calls"),
            )
            row.exchange_ts_ns >= encoder.previous_exchange_ts_ns ||
                throw(ArgumentError("encoded feature timestamps must not decrease across calls"))
            for channel in eachindex(current)
                scaled = (current[channel] - encoder.previous[channel]) * encoder.scale
                magnitude = abs(scaled)
                if !isfinite(magnitude) || magnitude >= encoder.saturation
                    count = Int(encoder.saturation)
                elseif magnitude >= encoder.threshold
                    count = max(1, floor(Int, magnitude))
                else
                    count = 0
                end
                if count > 0
                    spikes[index, channel] = Int8(sign(scaled) * count)
                end
            end
        end
        encoder.previous = current
        encoder.previous_exchange_ts_ns = row.exchange_ts_ns
        encoder.previous_sequence = row.sequence
    end
    return SpikeFrame(
        [row.exchange_ts_ns for row in frame],
        [row.sequence for row in frame],
        spikes,
    )
end

_spike_features(row::FeatureRow) = Float64[
    row.spread_ticks,
    row.mid_ticks,
    row.microprice_ticks,
    row.imbalance,
    row.signed_order_flow,
]
