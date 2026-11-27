# SPDX-License-Identifier: MIT OR Apache-2.0

const EXPERIMENT_CONFIG_SCHEMA = "dendritetrader.experiment.v1"

"""
    OutcomeHorizonSpec(name, kind; n, ns)

One outcome evaluation horizon for the trajectory export. `kind` is `:events`
(evaluate `n` venue events after the decision), `:wall_ns` (first ledger step at or
after `decision_ts + ns`), `:close` (first later step where inventory returns to
flat), or `:session_end` (final step). `n`/`ns` apply to `:events`/`:wall_ns` only.
"""
struct OutcomeHorizonSpec
    name::String
    kind::Symbol
    n::Union{Nothing, Int64}
    ns::Union{Nothing, Int64}
end

OutcomeHorizonSpec(name::AbstractString, kind::Symbol; n = nothing, ns = nothing) =
    OutcomeHorizonSpec(name, kind, n, ns)

"""
    ExperimentConfig

Validated experiment configuration for `run_experiment` (Task 6 / issue #76). This
is experiment TOML for a research **paper** run — dataset, splits, policy, latency,
feed, fill, fee, risk and outcome-horizon assumptions — and is distinct from the
`ExecutionEngine` TOML/YAML UX in `load_config` (GH#73/RM-366), which stays parked.
"""
struct ExperimentConfig
    name::String
    seed::UInt64
    session_path::String
    session_sha256::Union{Nothing, String}
    allow_sequence_gaps::Bool
    initial_cash_subunits::Int64
    portfolio_mode::Symbol
    scale::ParityScale
    latency::FixedLatency
    feed::AbstractFeedDelay
    fill_model::L2QueueApproximation
    fees::FeeSchedule
    limits::RiskLimits
    policy::Symbol
    model::Union{Nothing, Symbol}
    horizon_events::Int
    threshold::Float64
    imbalance_threshold::Float64
    ridge::Float64
    order_lots::Int64
    policy_max_inventory_lots::Int64
    label_threshold_ticks::Float64
    train_size::Int
    validation_size::Int
    test_size::Int
    embargo::Int
    evaluate_on::Symbol
    horizons::Vector{OutcomeHorizonSpec}
    source::Dict{String, Any}
    source_path::String
end

# ── TOML validation helpers ──────────────────────────────────────────────────

function _check_keys(table::AbstractDict, allowed, section::AbstractString)
    for key in keys(table)
        key isa AbstractString ||
            throw(ArgumentError("$section keys must be strings, got $(repr(key))"))
        key in allowed || throw(ArgumentError("unknown key `$key` in [$section]"))
    end
    return nothing
end

function _required_table(config::AbstractDict, name::AbstractString)
    table = get(config, name, nothing)
    table isa AbstractDict ||
        throw(ArgumentError("missing or invalid [$name] table in experiment config"))
    return table
end

function _req_int(table, key, section)
    value = get(table, key, nothing)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("[$section] `$key` must be an integer, got $(repr(value))"))
    return Int64(value)
end

function _opt_int(table, key, section, default)
    haskey(table, key) || return Int64(default)
    return _req_int(table, key, section)
end

function _req_string(table, key, section)
    value = get(table, key, nothing)
    value isa AbstractString ||
        throw(ArgumentError("[$section] `$key` must be a string, got $(repr(value))"))
    return String(value)
end

function _opt_string(table, key, section, default)
    haskey(table, key) || return default
    return _req_string(table, key, section)
end

function _opt_bool(table, key, section, default)
    haskey(table, key) || return Bool(default)
    value = table[key]
    value isa Bool || throw(ArgumentError("[$section] `$key` must be a Bool, got $(repr(value))"))
    return value
end

function _opt_number(table, key, section, default)
    haskey(table, key) || return Float64(default)
    value = table[key]
    value isa Real && !(value isa Bool) ||
        throw(ArgumentError("[$section] `$key` must be a number, got $(repr(value))"))
    converted = Float64(value)
    isfinite(converted) ||
        throw(ArgumentError("[$section] `$key` must be finite, got $(repr(value))"))
    return converted
end

_positive(i, name) = i > 0 || throw(ArgumentError("$name must be positive, got $i"))
_nonneg(i, name) = i >= 0 || throw(ArgumentError("$name must be non-negative, got $i"))

function _horizon_spec(table, index::Int)
    table isa AbstractDict ||
        throw(ArgumentError("[[outcomes.horizons]] entry $index must be a table"))
    _check_keys(table, ("name", "kind", "n", "ns"), "outcomes.horizons")
    kind = _req_string(table, "kind", "outcomes.horizons")
    name = _opt_string(table, "name", "outcomes.horizons", "")
    if kind == "events"
        n = _req_int(table, "n", "outcomes.horizons")
        _positive(n, "outcomes.horizons n")
        haskey(table, "ns") &&
            throw(ArgumentError("[[outcomes.horizons]] events entry takes `n`, not `ns`"))
        isempty(name) && (name = "events_$n")
        return OutcomeHorizonSpec(name, :events, n, nothing)
    elseif kind == "wall_ns"
        ns = _req_int(table, "ns", "outcomes.horizons")
        _positive(ns, "outcomes.horizons ns")
        haskey(table, "n") &&
            throw(ArgumentError("[[outcomes.horizons]] wall_ns entry takes `ns`, not `n`"))
        isempty(name) && (name = "wall_$(ns)ns")
        return OutcomeHorizonSpec(name, :wall_ns, nothing, ns)
    elseif kind in ("close", "session_end")
        (haskey(table, "n") || haskey(table, "ns")) &&
            throw(ArgumentError("[[outcomes.horizons]] $kind entry takes no n/ns"))
        isempty(name) && (name = kind)
        return OutcomeHorizonSpec(name, Symbol(kind), nothing, nothing)
    end
    throw(
        ArgumentError(
            "[[outcomes.horizons]] kind must be events, wall_ns, close or session_end, got $(repr(kind))",
        ),
    )
end

"""
    load_experiment_config(path) -> ExperimentConfig

Load and validate a `dendritetrader.experiment.v1` TOML file. Every section and key
is checked; unknown keys, missing required values and wrong types raise
`ArgumentError`. `data.session` is resolved relative to the configuration file.
`seed` is required — an undeclared seed is a validation error, matching the Task 6
config checks. `splits.embargo` defaults to `policy.horizon_events` and must be at
least the declared forecast horizon.
"""
function load_experiment_config(path::AbstractString)
    lowercase(splitext(path)[2]) == ".toml" ||
        throw(ArgumentError("experiment configuration must be a .toml file, got $(repr(path))"))
    config = TOML.parsefile(path)
    config isa AbstractDict ||
        throw(ArgumentError("experiment configuration root must be a mapping"))
    _check_keys(
        config,
        (
            "schema",
            "name",
            "seed",
            "data",
            "account",
            "latency",
            "feed",
            "fill",
            "fees",
            "risk",
            "policy",
            "splits",
            "outcomes",
        ),
        "experiment",
    )

    schema = _req_string(config, "schema", "experiment")
    schema == EXPERIMENT_CONFIG_SCHEMA || throw(
        ArgumentError("experiment schema must be `$EXPERIMENT_CONFIG_SCHEMA`, got $(repr(schema))"),
    )
    name = _req_string(config, "name", "experiment")
    haskey(config, "seed") ||
        throw(ArgumentError("`seed` must be declared in the experiment configuration"))
    seed_value = config["seed"]
    seed_value isa Integer && !(seed_value isa Bool) && seed_value >= 0 ||
        throw(ArgumentError("`seed` must be a non-negative integer"))
    seed = UInt64(seed_value)

    # ── data ──
    data = _required_table(config, "data")
    _check_keys(data, ("session", "sha256", "allow_sequence_gaps"), "data")
    session_rel = _req_string(data, "session", "data")
    session_path =
        isabspath(session_rel) ? session_rel :
        normpath(joinpath(dirname(abspath(path)), session_rel))
    pinned = haskey(data, "sha256") ? lowercase(_req_string(data, "sha256", "data")) : nothing
    allow_gaps = _opt_bool(data, "allow_sequence_gaps", "data", false)

    # ── account ──
    account = get(config, "account", Dict{String, Any}())
    account isa AbstractDict || throw(ArgumentError("[account] must be a table"))
    _check_keys(account, ("initial_cash_subunits", "mode", "price_scale", "qty_scale"), "account")
    initial_cash = _opt_int(account, "initial_cash_subunits", "account", 100_000)
    _nonneg(initial_cash, "account initial_cash_subunits")
    mode_str = _opt_string(account, "mode", "account", "long_flat")
    mode_str in ("long_flat", "signed") ||
        throw(ArgumentError("[account] `mode` must be long_flat or signed, got $(repr(mode_str))"))
    price_scale = _opt_int(account, "price_scale", "account", 100)
    qty_scale = _opt_int(account, "qty_scale", "account", 1)
    qty_scale == 1 || throw(
        ArgumentError("[account] `qty_scale` must be 1; the simulator requires exact notional"),
    )
    scale = ParityScale(price_scale, qty_scale)

    # ── latency ──
    latency_table = _required_table(config, "latency")
    _check_keys(latency_table, ("order_entry_ns", "cancel_ns", "report_ns"), "latency")
    order_entry = _req_int(latency_table, "order_entry_ns", "latency")
    _nonneg(order_entry, "latency order_entry_ns")
    cancel = _opt_int(latency_table, "cancel_ns", "latency", order_entry)
    report = _opt_int(latency_table, "report_ns", "latency", order_entry)
    _nonneg(cancel, "latency cancel_ns")
    _nonneg(report, "latency report_ns")
    latency = FixedLatency(order_entry; cancel_ns = cancel, report_ns = report)

    # ── feed ──
    feed_table = get(config, "feed", Dict{String, Any}())
    feed_table isa AbstractDict || throw(ArgumentError("[feed] must be a table"))
    _check_keys(feed_table, ("model", "delay_ns"), "feed")
    feed_model = _opt_string(feed_table, "model", "feed", "receive_timestamp")
    feed = if feed_model == "receive_timestamp"
        haskey(feed_table, "delay_ns") &&
            throw(ArgumentError("[feed] `delay_ns` applies only to fixed_exchange_delay"))
        ReceiveTimestampFeed()
    elseif feed_model == "fixed_exchange_delay"
        delay = _req_int(feed_table, "delay_ns", "feed")
        _nonneg(delay, "feed delay_ns")
        FixedFeedDelay(delay)
    else
        throw(
            ArgumentError(
                "[feed] `model` must be receive_timestamp or fixed_exchange_delay, got $(repr(feed_model))",
            ),
        )
    end

    # ── fill model ──
    fill = get(config, "fill", Dict{String, Any}())
    fill isa AbstractDict || throw(ArgumentError("[fill] must be a table"))
    _check_keys(fill, ("queue_position", "taker_slippage_ticks", "seed"), "fill")
    queue = _opt_string(fill, "queue_position", "fill", "back")
    slippage = _opt_int(fill, "taker_slippage_ticks", "fill", 0)
    fill_seed = _opt_int(fill, "seed", "fill", Int64(seed))
    fill_model = L2QueueApproximation(;
        queue_position = Symbol(queue),
        taker_slippage_ticks = slippage,
        seed = fill_seed,
    )

    # ── fees ──
    fees_table = get(config, "fees", Dict{String, Any}())
    fees_table isa AbstractDict || throw(ArgumentError("[fees] must be a table"))
    _check_keys(fees_table, ("maker_fee_ppm", "taker_fee_ppm", "per_fill_fee_subunits"), "fees")
    fees = FeeSchedule(;
        maker_fee_ppm = _opt_int(fees_table, "maker_fee_ppm", "fees", 0),
        taker_fee_ppm = _opt_int(fees_table, "taker_fee_ppm", "fees", 0),
        per_fill_fee_subunits = _opt_int(fees_table, "per_fill_fee_subunits", "fees", 0),
    )

    # ── risk ──
    risk = get(config, "risk", Dict{String, Any}())
    risk isa AbstractDict || throw(ArgumentError("[risk] must be a table"))
    _check_keys(
        risk,
        ("max_order_lots", "max_inventory_lots", "max_initial_cash_subunits", "max_open_orders"),
        "risk",
    )
    limits = RiskLimits(;
        max_order_lots = _opt_int(risk, "max_order_lots", "risk", 100),
        max_inventory_lots = _opt_int(risk, "max_inventory_lots", "risk", 100),
        max_initial_cash_subunits = _opt_int(
            risk,
            "max_initial_cash_subunits",
            "risk",
            1_000_000_000,
        ),
        max_open_orders = _opt_int(risk, "max_open_orders", "risk", 16),
    )

    # ── policy ──
    policy_table = _required_table(config, "policy")
    _check_keys(
        policy_table,
        (
            "name",
            "model",
            "horizon_events",
            "threshold",
            "imbalance_threshold",
            "ridge",
            "order_lots",
            "max_inventory_lots",
            "label_threshold_ticks",
        ),
        "policy",
    )
    policy_name = _req_string(policy_table, "name", "policy")
    policy_name in ("forecast", "none", "external") || throw(
        ArgumentError(
            "[policy] `name` must be forecast, none or external, got $(repr(policy_name))",
        ),
    )
    horizon = _opt_int(policy_table, "horizon_events", "policy", 1)
    _positive(horizon, "policy horizon_events")
    model_name = if haskey(policy_table, "model")
        model_str = _req_string(policy_table, "model", "policy")
        model_str in ("stationary", "imbalance", "ridge") || throw(
            ArgumentError(
                "[policy] `model` must be stationary, imbalance or ridge, got $(repr(model_str))",
            ),
        )
        Symbol(model_str)
    else
        nothing
    end
    policy_name == "forecast" &&
        model_name === nothing &&
        throw(ArgumentError("[policy] `model` is required when `name` is forecast"))
    policy_name != "forecast" &&
        model_name !== nothing &&
        throw(ArgumentError("[policy] `model` applies only when `name` is forecast"))
    threshold = _opt_number(policy_table, "threshold", "policy", 0.5)
    0 < threshold <= 1 || throw(ArgumentError("[policy] `threshold` must be in (0, 1]"))
    imbalance_threshold = _opt_number(policy_table, "imbalance_threshold", "policy", 0.0)
    imbalance_threshold >= 0 ||
        throw(ArgumentError("[policy] `imbalance_threshold` must be non-negative"))
    ridge = _opt_number(policy_table, "ridge", "policy", 1.0)
    ridge > 0 || throw(ArgumentError("[policy] `ridge` must be positive"))
    order_lots = _opt_int(policy_table, "order_lots", "policy", 1)
    _positive(order_lots, "policy order_lots")
    policy_cap = _opt_int(policy_table, "max_inventory_lots", "policy", 10)
    _positive(policy_cap, "policy max_inventory_lots")
    label_threshold = _opt_number(policy_table, "label_threshold_ticks", "policy", 0.0)
    label_threshold >= 0 ||
        throw(ArgumentError("[policy] `label_threshold_ticks` must be non-negative"))

    # ── splits ──
    splits = _required_table(config, "splits")
    _check_keys(
        splits,
        ("train_size", "validation_size", "test_size", "embargo", "evaluate_on"),
        "splits",
    )
    train_size = _req_int(splits, "train_size", "splits")
    validation_size = _req_int(splits, "validation_size", "splits")
    test_size = _req_int(splits, "test_size", "splits")
    _positive(train_size, "splits train_size")
    _positive(validation_size, "splits validation_size")
    _positive(test_size, "splits test_size")
    embargo = _opt_int(splits, "embargo", "splits", Int(horizon))
    embargo >= horizon || throw(
        ArgumentError(
            "[splits] `embargo` ($embargo) must be at least the forecast/label horizon ($horizon)",
        ),
    )
    eval_str = _opt_string(splits, "evaluate_on", "splits", "test")
    eval_str in ("test", "validation_test", "all") || throw(
        ArgumentError(
            "[splits] `evaluate_on` must be test, validation_test or all, got $(repr(eval_str))",
        ),
    )

    # ── outcomes ──
    outcome_specs = OutcomeHorizonSpec[]
    outcomes_table = get(config, "outcomes", Dict{String, Any}())
    outcomes_table isa AbstractDict || throw(ArgumentError("[outcomes] must be a table"))
    _check_keys(outcomes_table, ("horizons",), "outcomes")
    horizon_tables = get(outcomes_table, "horizons", nothing)
    if horizon_tables === nothing
        push!(outcome_specs, OutcomeHorizonSpec("session_end", :session_end, nothing, nothing))
    else
        horizon_tables isa AbstractVector ||
            throw(ArgumentError("[outcomes] `horizons` must be an array of tables"))
        for (index, table) in enumerate(horizon_tables)
            push!(outcome_specs, _horizon_spec(table, index))
        end
        names = [spec.name for spec in outcome_specs]
        length(unique(names)) == length(names) ||
            throw(ArgumentError("[[outcomes.horizons]] names must be unique"))
        isempty(outcome_specs) && throw(ArgumentError("[outcomes] `horizons` must not be empty"))
    end

    return ExperimentConfig(
        name,
        seed,
        session_path,
        pinned,
        allow_gaps,
        initial_cash,
        Symbol(mode_str),
        scale,
        latency,
        feed,
        fill_model,
        fees,
        limits,
        Symbol(policy_name),
        model_name,
        Int(horizon),
        threshold,
        imbalance_threshold,
        ridge,
        order_lots,
        policy_cap,
        label_threshold,
        Int(train_size),
        Int(validation_size),
        Int(test_size),
        Int(embargo),
        Symbol(eval_str),
        outcome_specs,
        config,
        String(path),
    )
end
