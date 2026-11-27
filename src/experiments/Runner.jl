# SPDX-License-Identifier: MIT OR Apache-2.0

# Task 6 experiment runner (DendriteTrader #76 / Linear RM-2118). One declared TOML
# configuration drives replay, train/validation/test split enforcement, optional
# baseline model fitting (train rows only), the Task 5 paper simulator, trajectory
# export, outcome labeling and the artifact set in `Artifacts.jl`.

"""
    TrajectoryCall

One decision-function invocation captured by `run_experiment`: the observable
market state, the signal the policy produced (if any), and the intents it
returned. `decision_id` matches the simulator's per-call counter and the
`:decision` `LedgerPoint` reference.
"""
struct TrajectoryCall
    decision_id::Int64
    decision_ts_ns::Int64
    info_sequence::Int64
    info_exchange_ts_ns::Int64
    event::MarketEvent
    snapshot::BookSnapshot
    features::Union{Nothing, FeatureRow}
    reported_inventory_lots::Int64
    reported_cash_subunits::Int64
    signal::Union{Nothing, Dict{String, Any}}
    intents::Vector{Union{OrderIntent, CancelIntent}}
end

"""
    ExperimentResult

Everything one `run_experiment` produced: the validated config, replay session and
causal snapshots, the chronological split and per-event phase tags, the
train-fitted normalizer and model, the `SimulationResult`, the captured decision
calls, and the derived trajectory / prediction / SNN rows plus metrics, manifest,
`run_id` and `core_hash`. `write_artifacts` serializes all of it.
"""
struct ExperimentResult
    config::ExperimentConfig
    session::ReplaySession
    snapshots::Vector{BookSnapshot}
    split::ChronologicalSplit
    phase::Vector{Symbol}
    eval_sequences::Vector{Int64}
    normalizer::RollingZScore
    model::Union{Nothing, AbstractForecastModel}
    simulation::SimulationResult
    calls::Vector{TrajectoryCall}
    trajectories::Vector{Dict{String, Any}}
    predictions::Vector{Dict{String, Any}}
    snn::Vector{Dict{String, Any}}
    metrics::Dict{String, Any}
    manifest::Dict{String, Any}
    run_id::String
    core_hash::String
    dataset_sha256::String
end

_canon(x) = Simulation.canonical_json(x)

function _build_model(config::ExperimentConfig)
    config.model === nothing && return nothing
    config.model == :stationary && return StationaryModel()
    config.model == :imbalance && return ImbalanceRule(config.imbalance_threshold)
    config.model == :ridge && return RidgeClassifier(config.ridge)
    throw(ArgumentError("unknown model $(config.model)"))
end

# Events between split ranges are embargo; events past the test range are not part
# of any split and are tagged `:unused` rather than mislabeled as embargo.
function _phase_tags(split::ChronologicalSplit, n::Int)
    phase = fill(:unused, n)
    for index in (last(split.train) + 1):(first(split.validation) - 1)
        phase[index] = :embargo
    end
    for index in (last(split.validation) + 1):(first(split.test) - 1)
        phase[index] = :embargo
    end
    for index in split.train
        phase[index] = :train
    end
    for index in split.validation
        phase[index] = :validation
    end
    for index in split.test
        phase[index] = :test
    end
    return phase
end

function _eval_sequences(events, split::ChronologicalSplit, evaluate_on::Symbol)
    evaluate_on == :all && return Set{Int64}(e.sequence for e in events)
    indices = Int[]
    evaluate_on in (:validation_test, :validation) && append!(indices, collect(split.validation))
    evaluate_on == :validation || append!(indices, collect(split.test))
    return Set{Int64}(events[index].sequence for index in indices)
end

function _git(args...)
    try
        cmd = Cmd(["git", "-C", dirname(@__DIR__), args...])
        return strip(read(cmd, String))
    catch
        return nothing
    end
end

# Dependency pins come from the package's own Manifest.toml when present (it is
# gitignored, so a repo without one reports `nothing` rather than failing).
function _dependency_provenance()
    manifest_path = joinpath(dirname(@__DIR__), "Manifest.toml")
    isfile(manifest_path) || return (nothing, nothing)
    manifest_sha = bytes2hex(sha256(read(manifest_path)))
    deps = try
        parsed = TOML.parsefile(manifest_path)
        Dict{String, Any}(
            name => get(first(entries), "version", nothing) for
            (name, entries) in get(parsed, "deps", Dict{String, Any}())
        )
    catch
        nothing
    end
    return (deps, manifest_sha)
end

function _code_provenance()
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    version = Base.pkgversion(parentmodule(@__MODULE__))
    deps, manifest_sha = _dependency_provenance()
    return Dict{String, Any}(
        "julia_version" => string(VERSION),
        "package" => "DendriteTrader",
        "package_version" => version === nothing ? nothing : string(version),
        "git_commit" => commit === nothing || isempty(commit) ? nothing : commit,
        "git_dirty" => dirty === nothing ? nothing : !isempty(dirty),
        "dependencies" => deps,
        "manifest_toml_sha256" => manifest_sha,
        "os" => String(Sys.KERNEL),
        "arch" => String(Sys.ARCH),
        "julia_threads" => Threads.nthreads(),
    )
end

function _zero_cost(fees::FeeSchedule, fill_model::L2QueueApproximation)
    return fees.maker_fee_ppm == 0 &&
           fees.taker_fee_ppm == 0 &&
           fees.per_fill_fee_subunits == 0 &&
           fill_model.taker_slippage_ticks == 0
end

function _outcome_aggregates(trajectories, horizons)
    aggregates = Dict{String, Any}()
    for spec in horizons
        outcomes = [
            row["outcome"]["horizons"][spec.name] for
            row in trajectories if haskey(row["outcome"]["horizons"], spec.name) &&
                row["outcome"]["horizons"][spec.name]["paper_pnl_subunits"] !== nothing
        ]
        isempty(outcomes) && continue
        pnls = Float64.(getindex.(outcomes, "paper_pnl_subunits"))
        maes = Float64[
            o["max_adverse_excursion_subunits"] for
            o in outcomes if o["max_adverse_excursion_subunits"] !== nothing
        ]
        mfes = Float64[
            o["max_favorable_excursion_subunits"] for
            o in outcomes if o["max_favorable_excursion_subunits"] !== nothing
        ]
        aggregates[spec.name] = Dict{String, Any}(
            "count" => length(pnls),
            "total_pnl_subunits" => _checked_i64(
                sum(Int128.(getindex.(outcomes, "paper_pnl_subunits"))),
                "total pnl",
            ),
            "mean_pnl_subunits" => sum(pnls) / length(pnls),
            "win_rate" => count(>(0), pnls) / length(pnls),
            "mean_adverse_excursion_subunits" =>
                isempty(maes) ? nothing : sum(maes) / length(maes),
            "mean_favorable_excursion_subunits" =>
                isempty(mfes) ? nothing : sum(mfes) / length(mfes),
            "session_end_fallbacks" =>
                count(o -> o["end_reason"] == "session_end_fallback", outcomes),
        )
    end
    return aggregates
end

"""
    run_experiment(config::ExperimentConfig; decide = nothing, signal = nothing)
        -> ExperimentResult

Run one declared experiment end to end:

1. Hash and load the replay session; `data.sha256` pins the exact bytes parsed.
2. Build the chronological train/embargo/validation/embargo/test split over event
   positions; `evaluate_on` selects where the policy may act: `test`,
   `validation_test`, or `all` — `all` lets it act on every event including train
   and embargo (an in-sample/widened diagnostic, never the default).
3. Fit the optional baseline model on **train** snapshots/labels only and the
   `RollingZScore` on train feature rows only — validation and test rows are never
   visible to either fit (CLAUDE.md rule 5).
4. Drive `simulate!` through the wrapped policy: the signal (forecast or external)
   is recorded on every call; intents are only forwarded on evaluation sequences.
5. Emit the trajectory / prediction / SNN rows and the manifest. `run_id` and
   `core_hash` cover the config, dataset and output hashes only — never wall-clock.

Outcome labels are deliberately forward-looking — that is their purpose as a
supervised dataset — so a row's `outcome` reflects run events after its phase
boundary. Consumers building train-only sets must honor each row's `phase` tag
(and ignore `:unused` rows past the test split).

`policy.name = "external"` requires `decide`; `signal` is an optional callable
`ctx -> Union{Nothing, AbstractDict}` recorded next to every call. Built-in
policies do not accept `decide`/`signal`.
"""
function run_experiment(config::ExperimentConfig; decide = nothing, signal = nothing)
    raw = read(config.session_path)
    dataset_sha = bytes2hex(sha256(raw))
    if config.session_sha256 !== nothing && config.session_sha256 != dataset_sha
        throw(
            ArgumentError(
                "data.sha256 mismatch: config pins $(config.session_sha256), file hashes to $dataset_sha",
            ),
        )
    end
    # Parse the same bytes the SHA-256 pin verified; reloading by path could pick
    # up a file swapped between hashing and parsing.
    session = load_session_jsonl(
        IOBuffer(raw);
        policy = ReplayPolicy(; allow_sequence_gaps = config.allow_sequence_gaps),
    )
    events = session.events
    n = length(events)
    split = chronological_split(
        n;
        train_size = config.train_size,
        validation_size = config.validation_size,
        test_size = config.test_size,
        embargo = config.embargo,
    )
    phase = _phase_tags(split, n)
    phase_by_sequence = Dict{Int64, Symbol}(events[index].sequence => phase[index] for index in 1:n)
    eval_set = _eval_sequences(events, split, config.evaluate_on)

    snapshots = replay!(OrderBookState(session.venue, session.instrument), session)
    frame = microstructure_features(snapshots; allow_sequence_gaps = config.allow_sequence_gaps)
    train_lo = events[first(split.train)].sequence
    train_hi = events[last(split.train)].sequence
    train_rows = [row for row in frame if train_lo <= row.sequence <= train_hi]

    normalizer = RollingZScore()
    if !isempty(train_rows)
        fit!(normalizer, FeatureFrame(train_rows))
    end

    model = _build_model(config)
    if model !== nothing
        isempty(train_rows) && throw(
            ArgumentError(
                "no complete-book feature rows inside the train split; cannot fit $(config.model)",
            ),
        )
        train_snapshots = snapshots[first(split.train):last(split.train)]
        labels = label_event_horizon(
            train_snapshots;
            horizon_events = config.horizon_events,
            threshold_ticks = config.label_threshold_ticks,
            allow_sequence_gaps = config.allow_sequence_gaps,
        )
        fit!(model, FeatureFrame(train_rows), labels)
    end

    horizon = ForecastHorizon(config.horizon_events)
    if config.policy == :forecast
        decide === nothing ||
            throw(ArgumentError("`decide` is not accepted for the built-in forecast policy"))
        signal === nothing ||
            throw(ArgumentError("`signal` is not accepted for the built-in forecast policy"))
    elseif config.policy == :external
        decide !== nothing || throw(ArgumentError("policy `external` requires a `decide` callable"))
    elseif config.policy == :none
        decide === nothing || throw(ArgumentError("`decide` is not accepted for policy `none`"))
        signal === nothing || throw(ArgumentError("`signal` is not accepted for policy `none`"))
    else
        throw(ArgumentError("unknown policy $(config.policy)"))
    end

    signal_fn = if config.policy == :forecast
        ctx -> begin
            ctx.features === nothing && return nothing
            forecast = only(predict!(model, FeatureFrame(FeatureRow[ctx.features]), horizon))
            return _forecast_dict(forecast, String(config.model))
        end
    elseif config.policy == :external && signal !== nothing
        ctx -> begin
            value = signal(ctx)
            value === nothing && return nothing
            value isa AbstractDict || throw(
                ArgumentError(
                    "`signal` must return nothing or an AbstractDict, got $(typeof(value))",
                ),
            )
            return deepcopy(Dict{String, Any}(String(k) => v for (k, v) in value))
        end
    else
        ctx -> nothing
    end

    decide_fn = if config.policy == :forecast
        ctx -> begin
            ctx.features === nothing && return nothing
            isempty(ctx.open_orders) || return nothing
            forecast = only(predict!(model, FeatureFrame(FeatureRow[ctx.features]), horizon))
            p_down, _, p_up = forecast.probabilities
            inventory = ctx.reported_inventory_lots
            if p_up >= config.threshold &&
               inventory + config.order_lots <= config.policy_max_inventory_lots
                return OrderIntent(
                    Buy,
                    config.order_lots,
                    first(first(ctx.snapshot.asks));
                    time_in_force = :ioc,
                    tag = "forecast_up",
                )
            elseif p_down >= config.threshold && inventory > 0
                return OrderIntent(
                    Sell,
                    inventory,
                    first(first(ctx.snapshot.bids));
                    time_in_force = :ioc,
                    tag = "forecast_down",
                )
            end
            return nothing
        end
    elseif config.policy == :external
        ctx -> decide(ctx)
    else
        ctx -> nothing
    end

    calls = TrajectoryCall[]
    function wrapped(ctx::DecisionContext)
        sig = signal_fn(ctx)
        intents =
            ctx.info_sequence in eval_set ? _normalize_intents(decide_fn(ctx)) :
            Union{OrderIntent, CancelIntent}[]
        push!(
            calls,
            TrajectoryCall(
                Int64(length(calls) + 1),
                ctx.decision_ts_ns,
                ctx.info_sequence,
                ctx.info_exchange_ts_ns,
                ctx.event,
                ctx.snapshot,
                ctx.features,
                ctx.reported_inventory_lots,
                ctx.reported_cash_subunits,
                sig,
                intents,
            ),
        )
        return intents
    end

    portfolio = PaperPortfolio(
        session.instrument,
        config.initial_cash_subunits;
        scale = config.scale,
        mode = config.portfolio_mode,
    )
    sim_config = SimulationConfig(;
        latency = config.latency,
        feed = config.feed,
        fill_model = config.fill_model,
        fees = config.fees,
        limits = config.limits,
    )
    result = simulate!(portfolio, session, wrapped; config = sim_config)

    # The simulator increments its decision counter once per wrapped call, so call i
    # pairs with the i-th `:decision` ledger point.
    decision_steps =
        [index for (index, point) in enumerate(result.ledger) if point.kind == :decision]
    length(decision_steps) == length(calls) ||
        throw(ErrorException("decision ledger and captured calls diverged"))
    records_by_decision = Dict{Int64, Vector{DecisionRecord}}()
    for record in result.decisions
        push!(get!(records_by_decision, record.decision_id, DecisionRecord[]), record)
    end
    realized_by_fill = Dict{String, Int64}(
        row.fill_id => row.realized_pnl_delta for row in result.audit if row.fill_id !== nothing
    )
    view = _outcome_view(result.ledger, result.fills, result.orders, realized_by_fill)
    placeholder_run_id = "run-pending"
    trajectories = Dict{String, Any}[]
    predictions = Dict{String, Any}[]
    snn = Dict{String, Any}[]
    for (index, call) in enumerate(calls)
        step = decision_steps[index]
        records = get(records_by_decision, call.decision_id, DecisionRecord[])
        call_phase = phase_by_sequence[call.info_sequence]
        push!(
            trajectories,
            _trajectory_row(
                call,
                call_phase,
                records,
                view,
                config.horizons,
                step,
                placeholder_run_id,
            ),
        )
        push!(predictions, _prediction_row(call, call_phase, records, placeholder_run_id))
        push!(
            snn,
            _snn_row(
                call,
                call_phase,
                records,
                view,
                config.horizons,
                step,
                normalizer,
                placeholder_run_id,
            ),
        )
    end

    # Deterministic identifiers over the deterministic core only. run_id comes from
    # the inputs and the simulation result hash; the trajectory artifact hash is then
    # taken over the final stamped rows (it cannot be a core input itself).
    core_hash = bytes2hex(
        sha256(
            _canon(
                Dict{String, Any}(
                    "schema" => EXPERIMENT_CONFIG_SCHEMA,
                    "config" => config.source,
                    "dataset_sha256" => dataset_sha,
                    "simulation_result_hash" => result.result_hash,
                ),
            ),
        ),
    )
    run_id = "run-" * first(core_hash, 16)
    for row in trajectories
        row["run_id"] = run_id
    end
    for row in predictions
        row["run_id"] = run_id
    end
    for row in snn
        row["run_id"] = run_id
    end
    # Hash exactly the bytes `write_artifacts` puts in trajectory.jsonl (the JSONL
    # payload, trailing newline included) so the manifest verifies the file.
    trajectory_sha = bytes2hex(sha256(_jsonl(trajectories)))

    metrics = _experiment_metrics(config, result, calls, trajectories, phase, eval_set, run_id)
    manifest = _experiment_manifest(
        config,
        session,
        split,
        eval_set,
        normalizer,
        model,
        result,
        dataset_sha,
        trajectory_sha,
        core_hash,
        run_id,
    )
    return ExperimentResult(
        config,
        session,
        snapshots,
        split,
        phase,
        sort!(collect(eval_set)),
        normalizer,
        model,
        result,
        calls,
        trajectories,
        predictions,
        snn,
        metrics,
        manifest,
        run_id,
        core_hash,
        dataset_sha,
    )
end

function _count_symbols(values)
    counts = Dict{String, Any}()
    for value in values
        key = String(value)
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function _experiment_metrics(config, result, calls, trajectories, phase, eval_set, run_id)
    statuses = _count_symbols(Symbol(row["outcome"]["status"]) for row in trajectories)
    diagnostics = Dict{String, Any}(
        "diagnostic_zero_latency" => is_zero_latency_diagnostic(result),
        "zero_latency_components" => zero_latency_components(config.latency),
        "diagnostic_zero_cost" => _zero_cost(config.fees, config.fill_model),
    )
    return Dict{String, Any}(
        "schema" => "dendritetrader.experiment_metrics.v1",
        "paper_only" => true,
        "run_id" => run_id,
        "name" => config.name,
        "events" => length(phase),
        "phases" => _count_symbols(phase),
        "decision_calls" => length(calls),
        "evaluated_calls" => count(call -> call.info_sequence in eval_set, calls),
        "decision_statuses" => statuses,
        "simulation" => Simulation._metrics_dict(result.metrics),
        "outcomes" => _outcome_aggregates(trajectories, config.horizons),
        "reconciliation_errors" => reconciliation_errors(result),
        "diagnostics" => diagnostics,
    )
end

function _experiment_manifest(
    config,
    session,
    split,
    eval_set,
    normalizer,
    model,
    result,
    dataset_sha,
    trajectory_sha,
    core_hash,
    run_id,
)
    events = session.events
    range_dict(range) = Dict{String, Any}(
        "indices" => [first(range), last(range)],
        "sequences" => [events[first(range)].sequence, events[last(range)].sequence],
        "exchange_ts_ns" =>
            [events[first(range)].exchange_ts_ns, events[last(range)].exchange_ts_ns],
        "size" => length(range),
    )
    normalization = if normalizer.fitted
        params = normalization_parameters(normalizer)
        params["fitted"] = true
        params["fit"] = "train split feature rows only"
        params
    else
        Dict{String, Any}(
            "fitted" => false,
            "reason" => "no complete-book feature rows inside the train split",
        )
    end
    model_dict =
        model === nothing ? nothing :
        Dict{String, Any}(
            "name" => String(config.model),
            "fitted" => model.fitted,
            "fitted_through_sequence" => model.fitted_through_sequence,
            "horizon_events" => model.horizon === nothing ? nothing : model.horizon.events,
            "fit_scope" => "train split only",
        )
    zero_latency = zero_latency_components(config.latency)
    return Dict{String, Any}(
        "schema" => "dendritetrader.experiment_manifest.v1",
        "paper_only" => true,
        "run_id" => run_id,
        "name" => config.name,
        "generated_utc" =>
            Dates.format(Dates.now(Dates.UTC), Dates.dateformat"yyyy-mm-ddTHH:MM:SSZ"),
        "code" => _code_provenance(),
        "config" => config.source,
        "config_path" => config.source_path,
        "dataset" => Dict{String, Any}(
            "path" => config.session_path,
            "sha256" => dataset_sha,
            "pinned_sha256" => config.session_sha256,
            "pin_verified" =>
                config.session_sha256 === nothing ? nothing : config.session_sha256 == dataset_sha,
            "venue" => String(session.venue),
            "instrument" => session.instrument,
            "n_events" => length(events),
            "first_exchange_ts_ns" => first(events).exchange_ts_ns,
            "last_exchange_ts_ns" => last(events).exchange_ts_ns,
            "allow_sequence_gaps" => config.allow_sequence_gaps,
        ),
        "seeds" => Dict{String, Any}(
            "experiment" => string(config.seed),
            "fill_model" => string(config.fill_model.seed),
        ),
        "splits" => Dict{String, Any}(
            "train" => range_dict(split.train),
            "validation" => range_dict(split.validation),
            "test" => range_dict(split.test),
            "embargo" => split.embargo,
            "evaluate_on" => String(config.evaluate_on),
            "eval_sequences" => sort!(collect(eval_set)),
            "note" => "indices are event positions; evaluate_on = \"all\" lets the policy act on train, embargo and validation rows as an in-sample diagnostic, and events past the test range are tagged \"unused\"",
        ),
        "normalization" => normalization,
        "model" => model_dict,
        "policy" => Dict{String, Any}(
            "name" => String(config.policy),
            "threshold" => config.threshold,
            "order_lots" => config.order_lots,
            "max_inventory_lots" => config.policy_max_inventory_lots,
        ),
        "outcome_horizons" => [
            Dict{String, Any}(
                "name" => spec.name,
                "kind" => String(spec.kind),
                "n" => spec.n,
                "ns" => spec.ns,
            ) for spec in config.horizons
        ],
        "assumptions" => result.assumptions,
        "diagnostics" => Dict{String, Any}(
            "diagnostic_zero_latency" => !isempty(zero_latency),
            "zero_latency_components" => zero_latency,
            "diagnostic_zero_cost" => _zero_cost(config.fees, config.fill_model),
        ),
        "reconciliation_errors" => reconciliation_errors(result),
        "hashes" => Dict{String, Any}(
            "core" => core_hash,
            "dataset_sha256" => dataset_sha,
            "simulation_result" => result.result_hash,
            "trajectory_sha256" => trajectory_sha,
        ),
        "artifacts" => Dict{String, Any}(),
        "reproduction" => Dict{String, Any}(
            "command" => "julia --project=. scripts/run_experiment.jl <config.toml> <outdir>",
            "determinism" => "core_hash covers the config document, dataset hash, simulation result hash and trajectory hash; generated_utc is outside it",
        ),
        "disclaimer" => "Experimental research output on replayed data. Fills are L2 approximations. Not a production HFT runtime, profit claim or financial advice.",
    )
end
