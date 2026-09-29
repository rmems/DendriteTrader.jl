# SPDX-License-Identifier: MIT OR Apache-2.0

"""A positive event-count forecast horizon."""
struct ForecastHorizon
    events::Int
    function ForecastHorizon(events::Integer)
        events > 0 || throw(ArgumentError("forecast horizon must be positive"))
        events <= typemax(Int) || throw(ArgumentError("forecast horizon must fit in Int"))
        new(Int(events))
    end
end

const FORECAST_CLASSES = (Down, Flat, Up)

"""
One normalized three-class forecast at a causal feature timestamp.

`probabilities` are class probabilities, not a calibrated win probability for
position sizing. Calibration belongs to a separately fitted validation-stage step.
"""
struct Forecast
    horizon::ForecastHorizon
    exchange_ts_ns::Int64
    sequence::Int64
    classes::NTuple{3, MovementLabel}
    probabilities::NTuple{3, Float64}
    function Forecast(
        horizon::ForecastHorizon,
        exchange_ts_ns::Integer,
        sequence::Integer,
        classes::NTuple{3, MovementLabel},
        probabilities,
    )
        exchange_ts_ns > 0 || throw(ArgumentError("forecast timestamp must be positive"))
        exchange_ts_ns <= typemax(Int64) ||
            throw(ArgumentError("forecast timestamp must fit in Int64"))
        sequence > 0 || throw(ArgumentError("forecast sequence must be positive"))
        sequence <= typemax(Int64) || throw(ArgumentError("forecast sequence must fit in Int64"))
        classes == FORECAST_CLASSES ||
            throw(ArgumentError("forecast classes must be ordered (Down, Flat, Up)"))
        length(probabilities) == 3 || throw(ArgumentError("forecast needs three probabilities"))
        probs = ntuple(i -> Float64(probabilities[i]), 3)
        all(isfinite, probs) || throw(ArgumentError("forecast probabilities must be finite"))
        all(>=(0.0), probs) || throw(ArgumentError("forecast probabilities must be non-negative"))
        isapprox(sum(probs), 1.0; atol = 1e-12, rtol = 1e-12) ||
            throw(ArgumentError("forecast probabilities must sum to one"))
        return new(horizon, Int64(exchange_ts_ns), Int64(sequence), classes, probs)
    end
end

Forecast(horizon::ForecastHorizon, row::FeatureRow, probabilities) =
    Forecast(horizon, row.exchange_ts_ns, row.sequence, FORECAST_CLASSES, probabilities)

abstract type AbstractForecastModel end

function fit!(::AbstractForecastModel, ::FeatureFrame, ::AbstractVector{MovementTarget})
    throw(MethodError(fit!, (AbstractForecastModel, FeatureFrame, AbstractVector{MovementTarget})))
end

function predict!(::AbstractForecastModel, ::FeatureFrame, ::ForecastHorizon)
    throw(MethodError(predict!, (AbstractForecastModel, FeatureFrame, ForecastHorizon)))
end

"""Reset transient inference state. Stateless baselines intentionally leave fitted weights untouched."""
reset_state!(model::AbstractForecastModel) = model
