# SPDX-License-Identifier: MIT OR Apache-2.0

@testset "Forecast contracts and CPU baselines" begin
    rows = FeatureFrame([FeatureRow(i * 10, i, 1, i, i, i - 2, 0) for i in 1:4])
    targets =
        [MovementTarget(1, 2, 1, Down), MovementTarget(2, 3, 1, Flat), MovementTarget(3, 4, 1, Up)]
    horizon = ForecastHorizon(1)
    @test_throws ArgumentError ForecastHorizon(0)
    @test_throws ArgumentError Forecast(horizon, 1, 1, (Up, Flat, Down), (0.0, 0.0, 1.0))
    @test_throws ArgumentError Forecast(horizon, 1, 1, (Down, Flat, Up), (0.2, 0.2, 0.2))
    @test_throws ArgumentError Forecast(horizon, 1, 1, (Down, Flat, Up), (NaN, 0.0, 1.0))
    @test_throws ArgumentError Forecast(
        horizon,
        typemax(UInt64),
        1,
        (Down, Flat, Up),
        (1.0, 0.0, 0.0),
    )
    @test_throws ArgumentError Forecast(
        horizon,
        1,
        typemax(UInt64),
        (Down, Flat, Up),
        (1.0, 0.0, 0.0),
    )

    for model in (StationaryModel(), ImbalanceRule(), RidgeClassifier())
        @test_throws ArgumentError predict!(model, rows, horizon)
        fit!(model, rows, targets)
        before = predict!(model, rows, horizon)
        @test reset_state!(model) === model
        @test predict!(model, rows, horizon) == before
        @test all(sum(f.probabilities) ≈ 1.0 for f in before)
        @test_throws ArgumentError predict!(model, rows, ForecastHorizon(2))
    end
    stationary = fit!(StationaryModel(), rows, targets)
    held_out = FeatureFrame([FeatureRow(50, 5, 1, 1, 1, 1e9, 1)])
    state = (stationary.probabilities, stationary.fitted_through_sequence)
    predict!(stationary, held_out, horizon)
    @test (stationary.probabilities, stationary.fitted_through_sequence) == state
    @test_throws ArgumentError fit!(StationaryModel(), rows, MovementTarget[])
    @test_throws ArgumentError fit!(StationaryModel(), rows, [MovementTarget(1, 1, 1, Up)])
    @test_throws ArgumentError fit!(
        StationaryModel(),
        rows,
        [MovementTarget(1, 2, 1, Up), MovementTarget(2, 3, 2, Flat)],
    )

    lookahead = fit!(
        StationaryModel(),
        rows,
        [MovementTarget(1, 2, 1, Down), MovementTarget(2, 3, 1, Flat), MovementTarget(3, 5, 1, Up)],
    )
    @test lookahead.probabilities == (0.5, 0.5, 0.0)
    @test lookahead.fitted_through_sequence == 3
    @test_throws ArgumentError fit!(
        StationaryModel(),
        rows,
        [MovementTarget(1, 5, 1, Up), MovementTarget(2, 6, 1, Flat)],
    )

    zero_variance = FeatureFrame([FeatureRow(i * 10, i, 1, 1, 1, 0, 0) for i in 1:3])
    tiny_ridge = fit!(RidgeClassifier(1e-20), zero_variance, targets)
    @test all(
        forecast -> all(isfinite, forecast.probabilities),
        predict!(tiny_ridge, zero_variance, horizon),
    )
    ridge = fit!(RidgeClassifier(), zero_variance, targets)
    @test all(
        forecast -> all(isfinite, forecast.probabilities),
        predict!(ridge, zero_variance, horizon),
    )
    extreme = FeatureFrame([
        FeatureRow(i * 10, i, floatmax(Float64), floatmax(Float64), floatmax(Float64), 0, 0) for
        i in 1:3
    ])
    @test all(
        forecast -> all(isfinite, forecast.probabilities),
        predict!(fit!(RidgeClassifier(), extreme, targets), extreme, horizon),
    )

    denormal_scale = FeatureFrame([
        FeatureRow(i * 10, i, nextfloat(0.0), nextfloat(0.0), nextfloat(0.0), 0, 0) for i in 1:3
    ])
    denormal_fit = fit!(RidgeClassifier(), denormal_scale, targets)
    @test all(
        forecast -> all(isfinite, forecast.probabilities),
        predict!(denormal_fit, extreme, horizon),
    )
end
