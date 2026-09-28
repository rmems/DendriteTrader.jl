# SPDX-License-Identifier: MIT OR Apache-2.0

abstract type AbstractSpikeEncoder end

"""Signed quantized spikes, one row per feature timestamp and one column per input channel."""
struct SpikeFrame
    exchange_ts_ns::Vector{Int64}
    sequence::Vector{Int64}
    spikes::Matrix{Int8}
    function SpikeFrame(
        exchange_ts_ns::AbstractVector{<:Integer},
        sequence::AbstractVector{<:Integer},
        spikes::AbstractMatrix{<:Integer},
    )
        length(exchange_ts_ns) == length(sequence) == size(spikes, 1) ||
            throw(ArgumentError("spike timestamps, sequences, and rows must agree"))
        size(spikes, 2) > 0 || throw(ArgumentError("spike frame must have at least one channel"))
        ts, seq = Int64.(exchange_ts_ns), Int64.(sequence)
        all(>(0), ts) && all(>(0), seq) ||
            throw(ArgumentError("spike timestamps and sequences must be positive"))
        all(diff(seq) .> 0) || throw(ArgumentError("spike sequences must strictly increase"))
        all(diff(ts) .>= 0) || throw(ArgumentError("spike timestamps must not decrease"))
        all(value -> typemin(Int8) <= value <= typemax(Int8), spikes) ||
            throw(ArgumentError("spikes must fit Int8"))
        new(ts, seq, Int8.(spikes))
    end
end

"""Return the fraction of non-silent channel observations in a `SpikeFrame`."""
spike_density(frame::SpikeFrame) =
    isempty(frame.spikes) ? 0.0 :
    count(value -> !iszero(value), frame.spikes) / length(frame.spikes)

"""Encode causal feature rows into a signed `SpikeFrame`, updating encoder state."""
function encode!(::AbstractSpikeEncoder, ::FeatureFrame)
    throw(MethodError(encode!, (AbstractSpikeEncoder, FeatureFrame)))
end

reset_state!(encoder::AbstractSpikeEncoder) = encoder
