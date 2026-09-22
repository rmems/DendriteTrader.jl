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
    return setprecision(BigFloat, 256) do
        count_big = BigFloat(length(training))
        means_big = ntuple(5) do index
            mean_big = sum(BigFloat(_feature_values(row)[index]) for row in training) / count_big
            isfinite(mean_big) && abs(mean_big) <= floatmax(Float64) ||
                throw(ArgumentError("training means must be finite"))
            mean_big
        end
        means = ntuple(index -> Float64(means_big[index]), 5)
        all(isfinite, means) || throw(ArgumentError("training means must be finite"))
        scales = ntuple(5) do index
            maximum_magnitude_big =
                maximum(abs(BigFloat(_feature_values(row)[index])) for row in training)
            isfinite(maximum_magnitude_big) && abs(maximum_magnitude_big) <= floatmax(Float64) ||
                throw(ArgumentError("training scales must be finite"))
            iszero(maximum_magnitude_big) && return 1.0
            scaled_mean = means_big[index] / maximum_magnitude_big
            variance_big =
                sum(
                    (BigFloat(_feature_values(row)[index]) / maximum_magnitude_big - scaled_mean)^2 for row in training
                ) / count_big
            iszero(variance_big) && return 1.0
            scale_big = maximum_magnitude_big * sqrt(variance_big)
            isfinite(scale_big) && abs(scale_big) <= floatmax(Float64) ||
                throw(ArgumentError("training scales must be finite"))
            scale = Float64(scale_big)
            return iszero(scale) ? 1.0 : scale
        end
        normalizer.means = means
        normalizer.scales = scales
        normalizer.fitted = true
        normalizer.fitted_through_sequence = last(training).sequence
        return normalizer
    end
end

function _normalized_value(value::Float64, mean::Float64, scale::Float64)
    difference = value - mean
    isfinite(difference) && return difference / scale
    magnitude = max(abs(value), abs(mean), abs(scale))
    return (value / magnitude - mean / magnitude) / (scale / magnitude)
end

"""Transform a frame in place without updating the training-fitted parameters."""
function transform!(normalizer::RollingZScore, frame::FeatureFrame)
    normalizer.fitted || throw(ArgumentError("normalizer must be fitted before transform"))
    transformed = FeatureRow[]
    sizehint!(transformed, length(frame))
    for row in frame
        values = _feature_values(row)
        normalized = ntuple(
            feature -> _normalized_value(
                values[feature],
                normalizer.means[feature],
                normalizer.scales[feature],
            ),
            5,
        )
        push!(transformed, FeatureRow(row.exchange_ts_ns, row.sequence, normalized...))
    end
    setfield!(frame, :rows, tuple(transformed...))
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
