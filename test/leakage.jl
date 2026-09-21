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

@testset "Normalization transform is transactional" begin
    frame = FeatureFrame([feature_row(1, 1), feature_row(2, floatmax(Float64))])
    before = collect(frame)
    tiny_scales = ntuple(_ -> 1.0e-308, 5)
    normalizer = RollingZScore(ntuple(_ -> 0.0, 5), tiny_scales, true, 0)

    @test_throws ArgumentError transform!(normalizer, frame)
    @test collect(frame) == before
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

    @test labels_before[1].label == Up
    @test labels_after[1] == labels_before[1]
    @test labels_after[2].label != labels_before[2].label

    mixed_session = copy(snapshots)
    mixed_session[3] = BookSnapshot(:OTHER, "XYZ", 300, 310, 3, [50 => 10], [52 => 10])
    @test_throws ArgumentError label_event_horizon(mixed_session; horizon_events = 1)
    @test_throws ArgumentError microstructure_features(mixed_session)
end

@testset "Labels retain exact event offsets" begin
    complete_one = BookSnapshot(:SIM, "XYZ", 100, 110, 1, [100 => 10], [102 => 10])
    incomplete = BookSnapshot(:SIM, "XYZ", 200, 210, 2, [101 => 10], Pair{Int64, Int64}[])
    complete_three = BookSnapshot(:SIM, "XYZ", 300, 310, 3, [102 => 10], [104 => 10])

    @test isempty(
        label_event_horizon([complete_one, incomplete, complete_three]; horizon_events = 1),
    )

    gap = BookSnapshot(:SIM, "XYZ", 300, 310, 4, [102 => 10], [104 => 10])
    @test_throws ArgumentError label_event_horizon([complete_one, gap]; horizon_events = 1)
    allowed =
        label_event_horizon([complete_one, gap]; horizon_events = 1, allow_sequence_gaps = true)
    @test only(allowed).anchor_sequence == 1
    @test only(allowed).target_sequence == 4
end

@testset "Snapshot clocks are valid for direct snapshots" begin
    impossible = BookSnapshot(:SIM, "XYZ", 100, 50, 1, [100 => 10], [102 => 10])

    @test_throws ArgumentError microstructure_features([impossible])
end

@testset "Top-of-book arithmetic handles Int64 sizes" begin
    largest_size = typemax(Int64)
    snapshot = BookSnapshot(
        :SIM,
        "XYZ",
        100,
        110,
        1,
        [100 => largest_size],
        [102 => largest_size],
    )

    row = only(microstructure_features([snapshot]))
    @test row.imbalance == 0.0
    @test row.microprice_ticks == 101.0
end

@testset "Labels reject malformed top-of-book snapshots" begin
    malformed = BookSnapshot(:SIM, "XYZ", 100, 110, 1, [100 => 0], [102 => 10])
    target = BookSnapshot(:SIM, "XYZ", 200, 210, 2, [101 => 10], [103 => 10])

    @test_throws ArgumentError label_event_horizon([malformed, target]; horizon_events = 1)
end

@testset "Labels retain one tick at large Int64 prices" begin
    boundary = Int64(1) << 53
    anchor = BookSnapshot(:SIM, "XYZ", 100, 110, 1, [boundary => 10], [boundary => 10])
    target = BookSnapshot(
        :SIM,
        "XYZ",
        200,
        210,
        2,
        [boundary + 1 => 10],
        [boundary + 1 => 10],
    )

    @test only(label_event_horizon([anchor, target]; horizon_events = 1)).label == Up
end

@testset "Normalization handles finite large-magnitude training rows" begin
    training = FeatureFrame([feature_row(1, 1.0e200), feature_row(2, -1.0e200)])

    normalizer = fit!(RollingZScore(), training)

    @test normalizer.means == ntuple(_ -> 0.0, 5)
    @test normalizer.scales == ntuple(_ -> 1.0e200, 5)
end

@testset "Normalization transformation avoids redundant full-frame copies" begin
    rows = [feature_row(sequence, Float64(sequence)) for sequence in 1:10_000]
    normalizer = fit!(RollingZScore(), FeatureFrame(rows))

    transform!(normalizer, FeatureFrame(rows))
    allocated = @allocated transform!(normalizer, FeatureFrame(rows))

    @test allocated < 3_500_000
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
    @test_throws ArgumentError ChronologicalSplit(-2:0, 3:5, 8:10, 2)
    @test_throws ArgumentError walk_forward_splits(
        BigInt(typemax(Int)) + 1;
        train_size = 1,
        validation_size = 1,
        test_size = 1,
        embargo = 0,
        max_horizon = 0,
    )
end
