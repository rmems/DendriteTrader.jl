# SPDX-License-Identifier: MIT OR Apache-2.0

const _LABEL_INDEX = Dict(Down => 1, Flat => 2, Up => 3)

function _training_labels(frame::FeatureFrame, targets::AbstractVector{MovementTarget})
    isempty(frame) && throw(ArgumentError("training frame must not be empty"))
    isempty(targets) && throw(ArgumentError("training targets must not be empty"))
    labels = Dict{Int64, MovementTarget}()
    horizon = nothing
    for target in targets
        target.anchor_sequence < target.target_sequence ||
            throw(ArgumentError("target sequence must strictly follow its anchor"))
        target.horizon_events > 0 || throw(ArgumentError("target horizon must be positive"))
        isnothing(horizon) && (horizon = target.horizon_events)
        target.horizon_events == horizon ||
            throw(ArgumentError("training targets must have one forecast horizon"))
        haskey(labels, target.anchor_sequence) &&
            throw(ArgumentError("training targets must have unique anchor sequences"))
        labels[target.anchor_sequence] = target
    end
    matched = [(row, labels[row.sequence]) for row in frame if haskey(labels, row.sequence)]
    isempty(matched) && throw(ArgumentError("training targets do not align with feature sequences"))
    return matched
end

function _set_fit_boundary!(model, matched)
    model.fitted_through_sequence = maximum(target.target_sequence for (_, target) in matched)
    model.horizon = ForecastHorizon(only(unique(target.horizon_events for (_, target) in matched)))
    model.fitted = true
    return model
end

function _validate_prediction_horizon(model, horizon::ForecastHorizon)
    model.fitted || throw(ArgumentError("model must be fit before prediction"))
    horizon == model.horizon ||
        throw(ArgumentError("prediction horizon must match fitted target horizon"))
end

_feature_vector(row::FeatureRow) = Float64[
    row.spread_ticks,
    row.mid_ticks,
    row.microprice_ticks,
    row.imbalance,
    row.signed_order_flow,
]

function _softmax(scores::AbstractVector{<:Real})
    all(isfinite, scores) || throw(ArgumentError("model produced non-finite scores"))
    offset = maximum(scores)
    weights = exp.(scores .- offset)
    return weights ./ sum(weights)
end

"""Training-frequency baseline with no feature-dependent state."""
mutable struct StationaryModel <: AbstractForecastModel
    probabilities::NTuple{3, Float64}
    fitted_through_sequence::Int64
    horizon::Union{Nothing, ForecastHorizon}
    fitted::Bool
end
StationaryModel() = StationaryModel((1 / 3, 1 / 3, 1 / 3), 0, nothing, false)

function fit!(model::StationaryModel, frame::FeatureFrame, targets::AbstractVector{MovementTarget})
    matched = _training_labels(frame, targets)
    counts = zeros(Int, 3)
    for (_, target) in matched
        counts[_LABEL_INDEX[target.label]] += 1
    end
    model.probabilities = ntuple(i -> counts[i] / length(matched), 3)
    return _set_fit_boundary!(model, matched)
end

function predict!(model::StationaryModel, frame::FeatureFrame, horizon::ForecastHorizon)
    _validate_prediction_horizon(model, horizon)
    return [Forecast(horizon, row, model.probabilities) for row in frame]
end

"""Deterministic imbalance rule; only values strictly beyond ±threshold map to Up or Down."""
mutable struct ImbalanceRule <: AbstractForecastModel
    threshold::Float64
    fitted_through_sequence::Int64
    horizon::Union{Nothing, ForecastHorizon}
    fitted::Bool
    function ImbalanceRule(threshold::Real = 0.0)
        threshold_float = Float64(threshold)
        isfinite(threshold_float) && threshold_float >= 0 ||
            throw(ArgumentError("imbalance threshold must be finite and non-negative"))
        new(threshold_float, 0, nothing, false)
    end
end

function fit!(model::ImbalanceRule, frame::FeatureFrame, targets::AbstractVector{MovementTarget})
    matched = _training_labels(frame, targets)
    return _set_fit_boundary!(model, matched)
end
function predict!(model::ImbalanceRule, frame::FeatureFrame, horizon::ForecastHorizon)
    _validate_prediction_horizon(model, horizon)
    return [
        Forecast(
            horizon,
            row,
            row.imbalance > model.threshold ? (0.0, 0.0, 1.0) :
            row.imbalance < -model.threshold ? (1.0, 0.0, 0.0) : (0.0, 1.0, 0.0),
        ) for row in frame
    ]
end

"""Small deterministic one-vs-class ridge baseline using the five causal inputs."""
mutable struct RidgeClassifier <: AbstractForecastModel
    ridge::Float64
    weights::Matrix{Float64}
    scales::NTuple{5, Float64}
    fitted_through_sequence::Int64
    horizon::Union{Nothing, ForecastHorizon}
    fitted::Bool
    function RidgeClassifier(ridge::Real = 1.0)
        ridge_float = Float64(ridge)
        isfinite(ridge_float) && ridge_float > 0 ||
            throw(ArgumentError("ridge must be finite and positive"))
        new(ridge_float, zeros(6, 3), ntuple(_ -> 1.0, 5), 0, nothing, false)
    end
end
function fit!(model::RidgeClassifier, frame::FeatureFrame, targets::AbstractVector{MovementTarget})
    matched = _training_labels(frame, targets)
    n = length(matched)
    x = ones(Float64, n, 6)
    y = zeros(Float64, n, 3)
    model.scales = ntuple(
        channel -> begin
            scale = maximum(abs(_feature_vector(row)[channel]) for (row, _) in matched)
            scale == 0.0 ? 1.0 : scale
        end,
        5,
    )
    for (index, (row, target)) in enumerate(matched)
        x[index, 2:end] .= _feature_vector(row) ./ model.scales
        y[index, _LABEL_INDEX[target.label]] = 1.0
    end
    penalty = Diagonal(vcat(0.0, fill(model.ridge, 5)))
    model.weights = (x' * x + penalty) \ (x' * y)
    all(isfinite, model.weights) || throw(ArgumentError("ridge fit produced non-finite weights"))
    return _set_fit_boundary!(model, matched)
end
function predict!(model::RidgeClassifier, frame::FeatureFrame, horizon::ForecastHorizon)
    _validate_prediction_horizon(model, horizon)
    return [
        Forecast(
            horizon,
            row,
            _softmax(vec(vcat(1.0, _feature_vector(row) ./ model.scales)' * model.weights)),
        ) for row in frame
    ]
end
