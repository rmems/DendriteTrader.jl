# SPDX-License-Identifier: MIT OR Apache-2.0

module Models

using LinearAlgebra
using ..DendriteTrader: FeatureFrame, FeatureRow, MovementTarget, MovementLabel, Down, Flat, Up
import ..DendriteTrader: fit!, reset_state!

export ForecastHorizon, Forecast, AbstractForecastModel, fit!, predict!, reset_state!
export StationaryModel, ImbalanceRule, RidgeClassifier

include("Interface.jl")
include("Baselines.jl")

end # module
