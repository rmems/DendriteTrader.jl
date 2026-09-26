# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using JSON
using DendriteTrader

@testset "Causal feature API" begin
    required = (
        :FeatureRow,
        :FeatureFrame,
        :microstructure_features,
        :RollingZScore,
        :fit!,
        :transform!,
        :normalization_parameters,
        :MovementLabel,
        :MovementTarget,
        :Down,
        :Flat,
        :Up,
        :label_event_horizon,
        :ChronologicalSplit,
        :walk_forward_splits,
    )
    for name in required
        @test isdefined(DendriteTrader, name)
    end
end

@testset "Hand-derived microstructure features" begin
    session = load_session_jsonl(joinpath(@__DIR__, "fixtures", "book_session.jsonl"))
    snapshots = replay!(OrderBookState(:SIM, "XYZ"), session)
    frame = microstructure_features(snapshots)

    @test length(frame) == 5
    @test [row.sequence for row in frame] == collect(2:6)

    first_row = frame[1]
    @test first_row.spread_ticks == 2.0
    @test first_row.mid_ticks == 101.0
    @test first_row.microprice_ticks ≈ 2220 / 22
    @test first_row.imbalance ≈ -2 / 22
    @test first_row.signed_order_flow == 0.0

    @test frame[2].spread_ticks == 1.0
    @test frame[2].mid_ticks == 101.5
    @test frame[2].microprice_ticks ≈ 1926 / 19
    @test frame[2].imbalance ≈ -5 / 19
    @test [row.signed_order_flow for row in frame] == [0.0, 7.0, 0.0, -2.0, -5.0]
end

@testset "Microstructure features require contiguous exchange sequences by default" begin
    snapshots = BookSnapshot[
        BookSnapshot(:SIM, "XYZ", 100, 101, 1, [100 => 10], [102 => 12]),
        BookSnapshot(:SIM, "XYZ", 200, 201, 3, [100 => 12], [102 => 12]),
    ]

    @test_throws ArgumentError microstructure_features(snapshots)

    observed_events = microstructure_features(snapshots; allow_sequence_gaps = true)
    @test [row.sequence for row in observed_events] == [1, 3]
    @test observed_events[2].signed_order_flow == 2.0
end

@testset "Microstructure features reject nonpositive top prices" begin
    snapshots = [
        BookSnapshot(:SIM, "XYZ", 100, 110, 1, [-2 => 10], [-1 => 10]),
        BookSnapshot(:SIM, "XYZ", 200, 210, 2, [-2 => 10], [-1 => 10]),
    ]

    @test_throws ArgumentError microstructure_features(snapshots)
end

@testset "Signed order flow preserves single-unit changes above Float64 precision" begin
    large_size = Int64(1) << 53
    snapshots = [
        BookSnapshot(:SIM, "XYZ", 100, 110, 1, [100 => large_size], [102 => 10]),
        BookSnapshot(:SIM, "XYZ", 200, 210, 2, [100 => large_size + 1], [102 => 10]),
    ]

    frame = microstructure_features(snapshots)

    @test frame[2].signed_order_flow == 1.0
end

@testset "Movement labels use exact future event horizons" begin
    session = load_session_jsonl(joinpath(@__DIR__, "fixtures", "book_session.jsonl"))
    snapshots = replay!(OrderBookState(:SIM, "XYZ"), session)

    one_event = label_event_horizon(snapshots; horizon_events = 1)
    two_events = label_event_horizon(snapshots; horizon_events = 2)
    thresholded = label_event_horizon(snapshots; horizon_events = 1, threshold_ticks = 0.5)

    @test [target.label for target in one_event] == [Up, Flat, Flat, Down]
    @test [(target.anchor_sequence, target.target_sequence) for target in one_event] == [(2, 3), (3, 4), (4, 5), (5, 6)]
    @test [target.label for target in two_events] == [Up, Flat, Down]
    @test [target.label for target in thresholded] == [Flat, Flat, Flat, Flat]
    @test_throws ArgumentError label_event_horizon(snapshots; horizon_events = 0)
    @test_throws ArgumentError label_event_horizon(snapshots; horizon_events = 6)
end

@testset "Training-fitted normalization" begin
    rows = [FeatureRow(100, 1, 1, 1, 1, 1, 1), FeatureRow(200, 2, 3, 3, 3, 3, 3)]
    normalizer = RollingZScore()
    fit!(normalizer, FeatureFrame(rows))

    @test normalizer.means == (2.0, 2.0, 2.0, 2.0, 2.0)
    @test normalizer.scales == (1.0, 1.0, 1.0, 1.0, 1.0)
    @test normalizer.fitted_through_sequence == 2

    frame = FeatureFrame(copy(rows))
    transform!(normalizer, frame)
    @test frame[1].spread_ticks == -1.0
    @test frame[2].signed_order_flow == 1.0
    @test JSON.parse(JSON.json(normalization_parameters(normalizer)))["fitted_through_sequence"] ==
          2

    replacement = FeatureRow(300, 3, 0, 0, 0, 0, 0)
    @test_throws Base.CanonicalIndexError setindex!(frame, replacement, 1)
    @test_throws ArgumentError setproperty!(frame, :rows, (replacement, frame[2]))
end
