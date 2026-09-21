# SPDX-License-Identifier: MIT OR Apache-2.0

@enum MovementLabel begin
    Down = -1
    Flat = 0
    Up = 1
end

function _complete_mid_prices(snapshots::AbstractVector{BookSnapshot})
    observations = Tuple{Int64, Float64}[]
    for snapshot in snapshots
        levels = _best_levels(snapshot)
        isnothing(levels) && continue
        bid, ask = levels
        push!(observations, (snapshot.sequence, (Float64(first(bid)) + first(ask)) / 2))
    end
    return observations
end

"""
    label_event_horizon(snapshots; horizon_events, threshold_ticks=0)

Label each complete snapshot against exactly `horizon_events` later complete
snapshots. The trailing observations without a target are omitted.
"""
function label_event_horizon(
    snapshots::AbstractVector{BookSnapshot};
    horizon_events::Integer,
    threshold_ticks::Real = 0,
)
    _validate_snapshot_session(snapshots)
    horizon_events > 0 || throw(ArgumentError("horizon_events must be positive"))
    isfinite(threshold_ticks) && threshold_ticks >= 0 ||
        throw(ArgumentError("threshold_ticks must be finite and non-negative"))
    observations = _complete_mid_prices(snapshots)
    horizon_events < length(observations) ||
        throw(ArgumentError("horizon_events must be shorter than the usable session"))

    labels = MovementLabel[]
    sizehint!(labels, length(observations) - horizon_events)
    for index in 1:(length(observations) - horizon_events)
        movement = observations[index + horizon_events][2] - observations[index][2]
        label = movement > threshold_ticks ? Up : movement < -threshold_ticks ? Down : Flat
        push!(labels, label)
    end
    return labels
end
