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
    _validate_snapshot_session(snapshots)
    horizon_events > 0 || throw(ArgumentError("horizon_events must be positive"))
    isfinite(threshold_ticks) && threshold_ticks >= 0 ||
        throw(ArgumentError("threshold_ticks must be finite and non-negative"))
    horizon_events < length(snapshots) ||
        throw(ArgumentError("horizon_events must be shorter than the session"))
    if !allow_sequence_gaps
        for index in 2:length(snapshots)
            snapshots[index].sequence == snapshots[index - 1].sequence + 1 ||
                throw(ArgumentError("exact event horizons require contiguous sequences"))
        end
    end

    labels = MovementTarget[]
    sizehint!(labels, length(snapshots) - horizon_events)
    for index in 1:(length(snapshots) - horizon_events)
        anchor = snapshots[index]
        target = snapshots[index + horizon_events]
        anchor_levels = _best_levels(anchor)
        target_levels = _best_levels(target)
        (isnothing(anchor_levels) || isnothing(target_levels)) && continue
        anchor_bid, anchor_ask = anchor_levels
        target_bid, target_ask = target_levels
        anchor_mid = (Float64(first(anchor_bid)) + first(anchor_ask)) / 2
        target_mid = (Float64(first(target_bid)) + first(target_ask)) / 2
        movement = target_mid - anchor_mid
        label = movement > threshold_ticks ? Up : movement < -threshold_ticks ? Down : Flat
        push!(labels, MovementTarget(anchor.sequence, target.sequence, Int(horizon_events), label))
    end
    return labels
end
