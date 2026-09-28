# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Deterministic signed delta encoder. For each feature channel, `round(abs(delta) * scale)`
spikes are emitted after the scaled magnitude reaches `threshold`, capped at
`saturation`; the sign carries delta direction. Resetting drops the prior sample,
so the first encoded row is silent.
"""
mutable struct DeltaEncoder <: AbstractSpikeEncoder
    scale::Float64
    threshold::Float64
    saturation::Int8
    previous::Union{Nothing, Vector{Float64}}
    function DeltaEncoder(; scale::Real = 1.0, threshold::Real = 1.0, saturation::Integer = 1)
        isfinite(scale) && scale > 0 || throw(ArgumentError("scale must be finite and positive"))
        isfinite(threshold) && threshold >= 0 ||
            throw(ArgumentError("threshold must be finite and non-negative"))
        1 <= saturation <= typemax(Int8) || throw(ArgumentError("saturation must be in 1:127"))
        new(Float64(scale), Float64(threshold), Int8(saturation), nothing)
    end
end

function reset_state!(encoder::DeltaEncoder)
    encoder.previous = nothing
    return encoder
end

function encode!(encoder::DeltaEncoder, frame::FeatureFrame)
    spikes = zeros(Int8, length(frame), 5)
    for (index, row) in enumerate(frame)
        current = _spike_features(row)
        if !isnothing(encoder.previous)
            for channel in eachindex(current)
                scaled = (current[channel] - encoder.previous[channel]) * encoder.scale
                magnitude = abs(scaled)
                if magnitude >= encoder.threshold
                    count = min(Int(encoder.saturation), round(Int, magnitude))
                    spikes[index, channel] = Int8(sign(scaled) * count)
                end
            end
        end
        encoder.previous = current
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
