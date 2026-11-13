# SPDX-License-Identifier: MIT OR Apache-2.0

# Leakage canaries for walk-forward splits, label horizons, and label isolation from the
# decision path. Every canary is paired with a deliberately leaky control that must trip
# the same check, so a vacuous assertion cannot pass silently.

module LeakageSplitsLabelsTests

using Test
using DendriteTrader

# ── Shared helpers ───────────────────────────────────────────────────────────

const EMPTY_LEVELS = Pair{Int64, Int64}[]

"""Deterministic, varied, complete top-of-book snapshot at contiguous sequence `i`."""
function varied_snapshot(i::Integer; offset::Integer = 0)
    bid = 1_000 + offset + round(Int, 5 * sin(0.7 * i)) + i % 3
    ask = bid + 1 + i % 2
    return BookSnapshot(
        :SIM,
        "XYZ",
        100 * i,
        100 * i + 5,
        i,
        [bid => 10 + (7 * i) % 13],
        [ask => 5 + (11 * i) % 17],
    )
end

"""Same clock as `varied_snapshot`, but with an empty bid side (no feature row, no label)."""
incomplete_snapshot(i::Integer) =
    BookSnapshot(:SIM, "XYZ", 100 * i, 100 * i + 5, i, EMPTY_LEVELS, [2_000 => 1])

evaluation_windows(split) = (split.validation, split.test)

"""
Count train rows whose label target (in event sequence) lands at or beyond the first
event sequence of a later evaluation window. `target_of(row)` returns the target
sequence of a row, or `nothing` when the row has no label.
"""
function label_horizon_violations(splits, row_sequences, target_of)
    violations = Tuple{Int, Int}[]
    for (split_index, split) in enumerate(splits)
        for window in evaluation_windows(split)
            first_eval_sequence = row_sequences[first(window)]
            for row in split.train
                target = target_of(row)
                isnothing(target) && continue
                target >= first_eval_sequence && push!(violations, (split_index, row))
            end
        end
    end
    return violations
end

"""
Count evaluation rows whose feature lookback start (in event sequence) reaches back to
or before the last training event sequence. `lookback_of(row)` returns the earliest
event sequence the row's features read, or `nothing` when the row reads only itself.
"""
function feature_lookback_violations(splits, row_sequences, lookback_of)
    violations = Tuple{Int, Int}[]
    for (split_index, split) in enumerate(splits)
        last_train_sequence = row_sequences[last(split.train)]
        for window in evaluation_windows(split), row in window
            start = lookback_of(row)
            isnothing(start) && continue
            start <= last_train_sequence && push!(violations, (split_index, row))
        end
    end
    return violations
end

"""Deliberately leaky control: contiguous windows with no embargo at all."""
function leaky_no_embargo_splits(n; train_size, validation_size, test_size)
    splits = ChronologicalSplit[]
    train_end = train_size
    while train_end + validation_size + test_size <= n
        validation = (train_end + 1):(train_end + validation_size)
        test = (last(validation) + 1):(last(validation) + test_size)
        push!(splits, ChronologicalSplit(1:train_end, validation, test, 0))
        train_end = last(test)
    end
    return splits
end

# ── 1. Splits when rows do not map 1:1 to events ─────────────────────────────

@testset "walk_forward_splits leakage: FeatureFrame with incomplete snapshots" begin
    # Every fifth event has an empty bid side, so `microstructure_features` skips it and
    # feature rows are a strict subset of event sequences (row index != sequence).
    horizon = 3
    snapshots = [i % 5 == 0 ? incomplete_snapshot(i) : varied_snapshot(i) for i in 1:80]
    frame = microstructure_features(snapshots)
    labels = label_event_horizon(snapshots; horizon_events = horizon)
    row_sequences = [row.sequence for row in frame]
    row_timestamps = [row.exchange_ts_ns for row in frame]
    expected_rows = count(i -> i % 5 != 0, 1:length(snapshots))
    @test length(frame) == expected_rows
    @test expected_rows < length(snapshots)
    @test row_sequences != collect(1:length(frame))

    target_by_anchor = Dict(label.anchor_sequence => label.target_sequence for label in labels)
    target_of(row) = get(target_by_anchor, row_sequences[row], nothing)
    # Signed order flow reads the previous complete snapshot, i.e. the previous row.
    lookback_of(row) = row == 1 ? nothing : row_sequences[row - 1]
    # Label horizon is `horizon` events; feature lookback is one row. Declare the max.
    max_horizon = max(horizon, 1)

    splits = walk_forward_splits(
        length(frame);
        train_size = 20,
        validation_size = 8,
        test_size = 8,
        embargo = max_horizon,
        max_horizon = max_horizon,
    )
    @test length(splits) >= 2
    @test isempty(label_horizon_violations(splits, row_sequences, target_of))
    @test isempty(feature_lookback_violations(splits, row_sequences, lookback_of))

    # Same check by exchange time: a train label's target timestamp precedes every
    # evaluation row's timestamp.
    timestamp_by_sequence = Dict(s.sequence => s.exchange_ts_ns for s in snapshots)
    for split in splits, window in evaluation_windows(split), row in split.train
        target = target_of(row)
        isnothing(target) && continue
        @test timestamp_by_sequence[target] < row_timestamps[first(window)]
    end

    # Control: the same assertion trips on a split function with no embargo.
    leaky =
        leaky_no_embargo_splits(length(frame); train_size = 20, validation_size = 8, test_size = 8)
    @test !isempty(label_horizon_violations(leaky, row_sequences, target_of))
    @test !isempty(feature_lookback_violations(leaky, row_sequences, lookback_of))
end

@testset "walk_forward_splits leakage: several rows per event sequence" begin
    # A row table outside `FeatureFrame` (e.g. per-level or merged trade/book rows) can
    # carry several rows per exchange event. Three rows share each sequence here.
    rows_per_event = 3
    event_count = 40
    horizon = 2           # label horizon, in exchange events
    lookback = 2          # feature lookback, in exchange events
    row_sequences = repeat(1:event_count, inner = rows_per_event)
    n = length(row_sequences)
    target_of(row) =
        row_sequences[row] + horizon <= event_count ? row_sequences[row] + horizon : nothing
    # Rows whose lookback would start before the first event have no earlier events to
    # read; return `nothing` so a non-positive start cannot satisfy the violation check
    # by construction.
    lookback_of(row) =
        row_sequences[row] - lookback < first(row_sequences) ? nothing :
        row_sequences[row] - lookback

    split_kwargs = (train_size = 30, validation_size = 9, test_size = 9)

    # Calling the API as documented ("embargo must be at least the maximum feature or
    # label horizon") with the horizon declared in events. The embargo is applied in row
    # positions, so it covers only embargo / rows_per_event events.
    documented = walk_forward_splits(
        n;
        split_kwargs...,
        embargo = max(horizon, lookback),
        max_horizon = max(horizon, lookback),
    )
    # KNOWN GAP (do not fix in this PR): `walk_forward_splits` indexes row positions,
    # not event sequences, so an event-denominated horizon under-embargoes whenever more
    # than one row maps to an event. These assertions document the leak and will start
    # passing (and must be flipped to @test) once splits are sequence-aware or reject
    # duplicate sequences.
    label_violations = label_horizon_violations(documented, row_sequences, target_of)
    lookback_violations = feature_lookback_violations(documented, row_sequences, lookback_of)
    @test_broken isempty(label_violations)
    @test_broken isempty(lookback_violations)
    # The violations are real event overlaps, not lookback starts at or below zero: e.g.
    # split 1 trains through row 30 (sequence 10) and validates from row 33 (sequence 11),
    # so train row 25 (sequence 9) has a target at sequence 11 inside validation, and
    # validation row 33 (sequence 11) reads back to sequence 9 inside train.
    @test (1, 25) in label_violations
    @test (1, 33) in lookback_violations
    @test all(lookback_of(row) >= first(row_sequences) for (_, row) in lookback_violations)

    # Scaling the row embargo by rows-per-event removes the overlap, which confirms the
    # violation above is a unit mismatch rather than a test artifact.
    scaled = rows_per_event * max(horizon, lookback)
    scaled_splits = walk_forward_splits(n; split_kwargs..., embargo = scaled, max_horizon = scaled)
    @test isempty(label_horizon_violations(scaled_splits, row_sequences, target_of))
    @test isempty(feature_lookback_violations(scaled_splits, row_sequences, lookback_of))

    # Control: no embargo at all trips the same checks.
    leaky = leaky_no_embargo_splits(n; split_kwargs...)
    @test !isempty(label_horizon_violations(leaky, row_sequences, target_of))
    @test !isempty(feature_lookback_violations(leaky, row_sequences, lookback_of))
end

# ── 2. RidgeClassifier labels stay inside the training frame ────────────────

ridge_parameters(model) =
    (model.weights, model.scales, model.fitted_through_sequence, model.horizon, model.fitted)

@testset "RidgeClassifier fit ignores labels and rows past the training frame" begin
    horizon = 3
    train_rows = 30
    original = [varied_snapshot(i) for i in 1:60]
    # Rewrite every event after the training frame by shifting bid and ask up by the
    # same 500 ticks. Labels whose anchor and target both lie past the boundary keep
    # their relative movement and do not change. Only labels anchored inside the train
    # frame whose targets cross the boundary (the straddling labels, which are exactly
    # the leak surface) flip.
    altered = [i <= train_rows ? original[i] : varied_snapshot(i; offset = 500) for i in 1:60]

    frame_original = microstructure_features(original)
    frame_altered = microstructure_features(altered)
    labels_original = label_event_horizon(original; horizon_events = horizon)
    labels_altered = label_event_horizon(altered; horizon_events = horizon)
    train_original = FeatureFrame(collect(frame_original)[1:train_rows])
    train_altered = FeatureFrame(collect(frame_altered)[1:train_rows])
    boundary = last(train_original).sequence
    @test collect(train_original) == collect(train_altered)
    @test labels_original != labels_altered
    straddling = [l for l in labels_original if l.anchor_sequence <= boundary < l.target_sequence]
    @test length(straddling) == horizon
    # The rewrite must flip at least one label, or the parameter-equality checks below
    # would be vacuous, and every flipped label must be a straddling one.
    @test length(labels_original) == length(labels_altered)
    flipped = [
        original_label for
        (original_label, altered_label) in zip(labels_original, labels_altered) if
        original_label != altered_label
    ]
    @test all(
        o.anchor_sequence == a.anchor_sequence && o.target_sequence == a.target_sequence for
        (o, a) in zip(labels_original, labels_altered)
    )
    @test length(flipped) > 0
    @test all(l -> l in straddling, flipped)

    with_future = fit!(RidgeClassifier(), train_original, labels_original)
    with_altered_future = fit!(RidgeClassifier(), train_altered, labels_altered)
    train_only_labels = [l for l in labels_original if l.target_sequence <= boundary]
    without_future = fit!(RidgeClassifier(), train_original, train_only_labels)

    @test ridge_parameters(with_future) == ridge_parameters(without_future)
    @test ridge_parameters(with_future) == ridge_parameters(with_altered_future)
    @test with_future.fitted_through_sequence <= boundary

    matched = DendriteTrader.Models._training_labels(train_original, labels_original)
    @test all(target.target_sequence <= boundary for (_, target) in matched)
    @test all(row.sequence <= boundary for (row, _) in matched)
    @test length(matched) == length(train_only_labels)

    # Control: a caller that hands `fit!` a frame extending past the training boundary
    # (the train-only input is not enforced by the API) is caught by the same checks.
    leaky_frame_original = FeatureFrame(collect(frame_original)[1:(train_rows + horizon + 5)])
    leaky_frame_altered = FeatureFrame(collect(frame_altered)[1:(train_rows + horizon + 5)])
    leaky_original = fit!(RidgeClassifier(), leaky_frame_original, labels_original)
    leaky_altered = fit!(RidgeClassifier(), leaky_frame_altered, labels_altered)
    @test leaky_original.fitted_through_sequence > boundary
    @test ridge_parameters(leaky_original) != ridge_parameters(leaky_altered)
    leaky_matched = DendriteTrader.Models._training_labels(leaky_frame_original, labels_original)
    @test any(target.target_sequence > boundary for (_, target) in leaky_matched)
end

@testset "RidgeClassifier fitted per walk-forward split never reaches evaluation" begin
    horizon = 3
    snapshots = [i % 7 == 0 ? incomplete_snapshot(i) : varied_snapshot(i) for i in 1:120]
    frame = microstructure_features(snapshots)
    labels = label_event_horizon(snapshots; horizon_events = horizon)
    rows = collect(frame)
    splits = walk_forward_splits(
        length(frame);
        train_size = 30,
        validation_size = 10,
        test_size = 10,
        embargo = horizon,
        max_horizon = horizon,
    )
    @test length(splits) >= 2
    for split in splits
        model = fit!(RidgeClassifier(), FeatureFrame(rows[split.train]), labels)
        @test model.fitted_through_sequence <= rows[last(split.train)].sequence
        @test model.fitted_through_sequence < rows[first(split.validation)].sequence
    end

    # Control: fitting on train plus validation rows (a leaky caller) crosses into the
    # evaluation window and fails the same comparison.
    split = first(splits)
    leaky = fit!(RidgeClassifier(), FeatureFrame(rows[1:last(split.validation)]), labels)
    @test !(leaky.fitted_through_sequence < rows[first(split.validation)].sequence)
end

# ── 3. MovementTarget cannot reach the decision path ─────────────────────────

"""
True when type expression `T` names a subtype of `X` anywhere inside it: directly, as a
`Union` member, as a type parameter (e.g. `AbstractVector{MovementTarget}`), inside a
`Vararg`, or as a `TypeVar` bound of a `UnionAll` (e.g. `Vector{<:MovementTarget}`).
`Any` and `Union{}` do not count, so an untyped argument is not reported.
"""
function mentions_type(T, X::Type, seen = Base.IdSet{Any}())
    T in seen && return false
    push!(seen, T)
    if T isa TypeVar
        return mentions_type(T.lb, X, seen) || mentions_type(T.ub, X, seen)
    elseif T isa UnionAll
        return mentions_type(T.var, X, seen) || mentions_type(T.body, X, seen)
    elseif T isa Union
        return any(U -> mentions_type(U, X, seen), Base.uniontypes(T))
    elseif T isa Core.TypeofVararg
        return isdefined(T, :T) && mentions_type(T.T, X, seen)
    elseif T isa DataType
        T === Any && return false
        T <: X && return true
        return any(P -> mentions_type(P, X, seen), T.parameters)
    end
    return false
end

"""
True when any method of `f` declares a positional argument whose type mentions
`MovementTarget`. The first signature entry (`typeof(f)`) is skipped. Keyword argument
types are not part of method signatures, so they are covered by the behavioral
`MethodError` checks below rather than by this detector.
"""
accepts_movement_target(f) = any(methods(f)) do m
    sig = Base.unwrap_unionall(m.sig)
    any(P -> mentions_type(P, MovementTarget), sig.parameters[2:end])
end

"""True when any field type of `T` mentions `MovementTarget`."""
carries_movement_target(T::Type) = any(F -> mentions_type(F, MovementTarget), fieldtypes(T))

# Deliberately leaky decision-path function and row type used as detector controls.
leaky_predict(model, frame::FeatureFrame, targets::AbstractVector{MovementTarget}) = nothing
leaky_encode(frame::FeatureFrame, target::Union{Nothing, MovementTarget}) = nothing
leaky_size(targets::Vector{<:MovementTarget}, rest::Vararg{Int}) = nothing
leaky_varargs(rows::Vararg{MovementTarget}) = nothing
struct LeakyRow
    sequence::Int64
    target::MovementTarget
end
struct LeakyOptionalRow
    sequence::Int64
    target::Union{Nothing, Vector{MovementTarget}}
end

@testset "MovementTarget is unreachable from decision-path APIs" begin
    decision_path = (
        predict!,
        encode!,
        reset_state!,
        spike_density,
        microstructure_features,
        transform!,
        normalization_parameters,
        size_position,
        kelly_fraction,
        half_kelly,
        from_confidence,
        risk_tier,
        passes_gate,
        validate_signal,
        execute_signal!,
        TradeSignal,
    )
    for f in decision_path
        @test !accepts_movement_target(f)
    end

    for T in (FeatureRow, FeatureFrame, SpikeFrame, Forecast, ForecastHorizon, TradeSignal)
        @test !carries_movement_target(T)
    end
    # FeatureRow and SpikeFrame must not carry any label type at all.
    @test !any(F -> F <: MovementLabel, fieldtypes(FeatureRow))
    @test !any(F -> occursin("MovementLabel", string(F)), fieldtypes(SpikeFrame))

    targets = label_event_horizon([varied_snapshot(i) for i in 1:6]; horizon_events = 1)
    model = RidgeClassifier()
    @test !hasmethod(predict!, Tuple{RidgeClassifier, typeof(targets), ForecastHorizon})
    @test !hasmethod(predict!, Tuple{RidgeClassifier, FeatureFrame, typeof(targets)})
    @test !hasmethod(encode!, Tuple{DeltaEncoder, typeof(targets)})
    @test !hasmethod(transform!, Tuple{RollingZScore, typeof(targets)})
    @test_throws MethodError predict!(model, targets, ForecastHorizon(1))
    @test_throws MethodError encode!(DeltaEncoder(), targets)

    # Controls: the detector does fire on a function that takes targets, both the
    # intended consumer (`fit!` for models) and a deliberately leaky predictor.
    @test accepts_movement_target(fit!)
    @test accepts_movement_target(leaky_predict)
    @test accepts_movement_target(leaky_encode)
    @test accepts_movement_target(leaky_size)
    @test accepts_movement_target(leaky_varargs)
    @test carries_movement_target(LeakyOptionalRow)
    # Untyped (`Any`) arguments and unrelated parameterized types are not reported.
    @test !mentions_type(Any, MovementTarget)
    @test !mentions_type(Vector{MovementLabel}, MovementTarget)
    @test !mentions_type(Union{}, MovementTarget)
    @test carries_movement_target(LeakyRow)
end

end # module LeakageSplitsLabelsTests
