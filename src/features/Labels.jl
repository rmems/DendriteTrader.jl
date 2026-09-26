# SPDX-License-Identifier: MIT OR Apache-2.0

@enum MovementLabel begin
    Down = -1
    Flat = 0
    Up = 1
end

"""A movement label with explicit anchor and exact target event sequences."""
struct MovementTarget
    anchor_sequence::Int64
    target_sequence::Int64
    horizon_events::Int
    label::MovementLabel
end

function _movement_label(movement_twice::Int128, threshold_ticks::Integer)
    threshold_twice = BigInt(2) * BigInt(threshold_ticks)
    return movement_twice > threshold_twice ? Up : movement_twice < -threshold_twice ? Down : Flat
end

function _movement_label(movement_twice::Int128, threshold_ticks::Rational)
    movement_scaled = BigInt(movement_twice) * BigInt(denominator(threshold_ticks))
    threshold_scaled = BigInt(2) * BigInt(numerator(threshold_ticks))
    return movement_scaled > threshold_scaled ? Up : movement_scaled < -threshold_scaled ? Down : Flat
end

function _movement_label(movement_twice::Int128, threshold_ticks::Real)
    working_precision =
        threshold_ticks isa BigFloat ? max(256, precision(threshold_ticks)) : 256
    return setprecision(BigFloat, working_precision) do
        threshold_twice = 2 * BigFloat(threshold_ticks)
        movement_twice_big = BigFloat(movement_twice)
        movement_twice_big > threshold_twice ? Up :
        movement_twice_big < -threshold_twice ? Down : Flat
    end
end

"""
    label_event_horizon(snapshots; horizon_events, threshold_ticks=0)

Label complete snapshots against the snapshot exactly `horizon_events` later in the
original event stream. Incomplete anchors or targets and trailing observations are
omitted. Sequence gaps are rejected unless `allow_sequence_gaps=true` explicitly
defines the horizon in observed rather than exchange events.
"""
function label_event_horizon(
    snapshots::AbstractVector{BookSnapshot};
    horizon_events::Integer,
    threshold_ticks::Real = 0,
    allow_sequence_gaps::Bool = false,
)
    _validate_snapshot_session(snapshots; allow_sequence_gaps)
    horizon_events > 0 || throw(ArgumentError("horizon_events must be positive"))
    isfinite(threshold_ticks) && threshold_ticks >= 0 ||
        throw(ArgumentError("threshold_ticks must be finite and non-negative"))
    horizon_events < length(snapshots) ||
        throw(ArgumentError("horizon_events must be shorter than the session"))

    validated_levels = [_validated_top_of_book(snapshot) for snapshot in snapshots]
    labels = MovementTarget[]
    sizehint!(labels, length(snapshots) - horizon_events)
    for index in 1:(length(snapshots) - horizon_events)
        anchor = snapshots[index]
        target = snapshots[index + horizon_events]
        anchor_levels = validated_levels[index]
        target_levels = validated_levels[index + horizon_events]
        (isnothing(anchor_levels) || isnothing(target_levels)) && continue
        anchor_bid, anchor_ask = anchor_levels
        target_bid, target_ask = target_levels
        anchor_mid_twice = Int128(first(anchor_bid)) + Int128(first(anchor_ask))
        target_mid_twice = Int128(first(target_bid)) + Int128(first(target_ask))
        movement_twice = target_mid_twice - anchor_mid_twice
        label = _movement_label(movement_twice, threshold_ticks)
        push!(labels, MovementTarget(anchor.sequence, target.sequence, Int(horizon_events), label))
    end
    return labels
end
