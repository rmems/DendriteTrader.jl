# SPDX-License-Identifier: MIT OR Apache-2.0

"""Training-fitted population z-score parameters for the five canonical features."""
mutable struct RollingZScore
    means::NTuple{5, Float64}
    scales::NTuple{5, Float64}
    fitted::Bool
    fitted_through_sequence::Int64
end

RollingZScore() = RollingZScore(ntuple(_ -> 0.0, 5), ntuple(_ -> 1.0, 5), false, 0)

_feature_values(row::FeatureRow) =
    (row.spread_ticks, row.mid_ticks, row.microprice_ticks, row.imbalance, row.signed_order_flow)

"""Fit normalization parameters from the supplied training frame only."""
function fit!(normalizer::RollingZScore, training::FeatureFrame)
    isempty(training) && throw(ArgumentError("training frame must be non-empty"))
    _validate_feature_rows(training.rows)
    count = length(training)
    means = ntuple(index -> sum(_feature_values(row)[index] / count for row in training), 5)
    all(isfinite, means) || throw(ArgumentError("training means must be finite"))
    scales = ntuple(5) do index
        variance = sum((_feature_values(row)[index] - means[index])^2 for row in training) / count
        scale = sqrt(variance)
        isfinite(scale) || throw(ArgumentError("training scales must be finite"))
        return iszero(scale) ? 1.0 : scale
    end
    normalizer.means = means
    normalizer.scales = scales
    normalizer.fitted = true
    normalizer.fitted_through_sequence = last(training).sequence
    return normalizer
end

"""Transform a frame in place without updating the training-fitted parameters."""
function transform!(normalizer::RollingZScore, frame::FeatureFrame)
    normalizer.fitted || throw(ArgumentError("normalizer must be fitted before transform"))
    transformed = FeatureRow[]
    sizehint!(transformed, length(frame))
    for row in frame
        values = _feature_values(row)
        normalized = ntuple(
            feature ->
                (values[feature] - normalizer.means[feature]) / normalizer.scales[feature],
            5,
        )
        push!(transformed, FeatureRow(row.exchange_ts_ns, row.sequence, normalized...))
    end
    frame.rows = tuple(transformed...)
    return frame
end

"""Return JSON-serializable normalization evidence."""
function normalization_parameters(normalizer::RollingZScore)
    normalizer.fitted || throw(ArgumentError("normalizer has not been fitted"))
    return Dict(
        "feature_names" => collect(FEATURE_NAMES),
        "means" => collect(normalizer.means),
        "scales" => collect(normalizer.scales),
        "fitted_through_sequence" => normalizer.fitted_through_sequence,
    )
end
