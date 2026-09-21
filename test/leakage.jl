# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using DendriteTrader

function feature_row(sequence, value)
    FeatureRow(sequence * 10, sequence, value, value, value, value, value)
end

@testset "Normalization leakage canary" begin
    training = FeatureFrame([feature_row(1, 1), feature_row(2, 3)])
    validation = FeatureFrame([feature_row(3, 1_000_000)])
    normalizer = fit!(RollingZScore(), training)
    parameters_before = normalization_parameters(normalizer)

    transform!(normalizer, validation)

    @test normalization_parameters(normalizer) == parameters_before
    @test validation[1].spread_ticks == 999_998.0
end

@testset "Label horizon leakage canary" begin
    snapshots = [
        BookSnapshot(:SIM, "XYZ", 100, 110, 1, [100 => 10], [102 => 10]),
        BookSnapshot(:SIM, "XYZ", 200, 210, 2, [101 => 10], [103 => 10]),
        BookSnapshot(:SIM, "XYZ", 300, 310, 3, [200 => 10], [202 => 10]),
    ]
    labels_before = label_event_horizon(snapshots; horizon_events = 1)
    snapshots[3] = BookSnapshot(:SIM, "XYZ", 300, 310, 3, [50 => 10], [52 => 10])
    labels_after = label_event_horizon(snapshots; horizon_events = 1)

    @test labels_before[1] == Up
    @test labels_after[1] == labels_before[1]
    @test labels_after[2] != labels_before[2]

    mixed_session = copy(snapshots)
    mixed_session[3] = BookSnapshot(:OTHER, "XYZ", 300, 310, 3, [50 => 10], [52 => 10])
    @test_throws ArgumentError label_event_horizon(mixed_session; horizon_events = 1)
    @test_throws ArgumentError microstructure_features(mixed_session)
end

@testset "Embargoed chronological walk-forward splits" begin
    splits = walk_forward_splits(
        52;
        train_size = 10,
        validation_size = 5,
        test_size = 5,
        embargo = 2,
        max_horizon = 2,
    )

    @test length(splits) == 3
    @test splits[1].train == 1:10
    @test splits[1].validation == 13:17
    @test splits[1].test == 20:24
    @test splits[2].train == 1:24
    @test splits[2].validation == 27:31
    @test splits[2].test == 34:38
    for split in splits
        @test isempty(intersect(split.train, split.validation))
        @test isempty(intersect(split.validation, split.test))
        @test first(split.validation) - last(split.train) - 1 == split.embargo
        @test first(split.test) - last(split.validation) - 1 == split.embargo
    end

    @test_throws ArgumentError walk_forward_splits(
        50;
        train_size = 10,
        validation_size = 5,
        test_size = 5,
        embargo = 1,
        max_horizon = 2,
    )
    @test_throws ArgumentError walk_forward_splits(
        10;
        train_size = 10,
        validation_size = 5,
        test_size = 5,
        embargo = 2,
        max_horizon = 2,
    )
end
