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

@testset "Movement labels use exact future event horizons" begin
    session = load_session_jsonl(joinpath(@__DIR__, "fixtures", "book_session.jsonl"))
    snapshots = replay!(OrderBookState(:SIM, "XYZ"), session)

    @test label_event_horizon(snapshots; horizon_events = 1) == [Up, Flat, Flat, Down]
    @test label_event_horizon(snapshots; horizon_events = 2) == [Up, Flat, Down]
    @test label_event_horizon(snapshots; horizon_events = 1, threshold_ticks = 0.5) ==
          [Flat, Flat, Flat, Flat]
    @test_throws ArgumentError label_event_horizon(snapshots; horizon_events = 0)
    @test_throws ArgumentError label_event_horizon(snapshots; horizon_events = 5)
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
end
