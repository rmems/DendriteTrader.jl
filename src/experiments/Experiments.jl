# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    Experiments

Research experiment tooling: chronological train/validation/test splits
(`ChronologicalSplit`, `walk_forward_splits`, `chronological_split`) and the Task 6
experiment runner (`load_experiment_config`, `run_experiment`, `write_artifacts`)
that produces the issue #76 paper-trading trajectory export, outcome labels,
manifest and SNN-ready rows. Included after `Simulation` because it drives it.
"""
module Experiments

using Dates
using JSON
using SHA
using TOML
using ..DendriteTrader: Buy, Sell, fit!, Simulation
using ..DendriteTrader.Market:
    MarketEvent,
    BookSnapshot,
    OrderBookState,
    ReplayPolicy,
    ReplaySession,
    replay!,
    load_session_jsonl
using ..DendriteTrader.Features:
    FeatureRow,
    FeatureFrame,
    RollingZScore,
    microstructure_features,
    label_event_horizon,
    transform!,
    normalization_parameters
using ..DendriteTrader.Models:
    AbstractForecastModel,
    ForecastHorizon,
    Forecast,
    predict!,
    StationaryModel,
    ImbalanceRule,
    RidgeClassifier
using ..DendriteTrader.Simulation:
    AbstractFeedDelay,
    CancelIntent,
    DecisionContext,
    DecisionRecord,
    FeeSchedule,
    Fill,
    FixedLatency,
    FixedFeedDelay,
    L2QueueApproximation,
    LedgerPoint,
    OrderIntent,
    PaperOrder,
    PaperPortfolio,
    ParityScale,
    ReceiveTimestampFeed,
    RiskLimits,
    SimulationConfig,
    SimulationResult,
    simulate!,
    reconciliation_errors,
    zero_latency_components,
    is_zero_latency_diagnostic

export ChronologicalSplit, walk_forward_splits, chronological_split
export OutcomeHorizonSpec, ExperimentConfig, load_experiment_config
export TrajectoryCall, ExperimentResult, run_experiment, write_artifacts

include("Splits.jl")
include("Config.jl")
include("Runner.jl")
include("Trajectory.jl")
include("Artifacts.jl")

end # module
