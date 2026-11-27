# SPDX-License-Identifier: MIT OR Apache-2.0

# Trajectory export for neuromorphic supervisor training (DendriteTrader #76 /
# Linear RM-2118, Task 6 of docs/superpowers/plans/2026-09-20-snn-hft-experiments.md).
# Every row preserves `market state -> neural signal -> gate/sizing decision ->
# paper action -> later outcome`, phase-tagged by the experiment split.

const TRAJECTORY_RECORD_TYPE = "paper_trajectory_step"
const SNN_RECORD_TYPE = "snn_trajectory_step"
const PREDICTION_RECORD_TYPE = "experiment_prediction"

function _features_dict(row::FeatureRow)
    return Dict{String, Any}(
        "spread_ticks" => row.spread_ticks,
        "mid_ticks" => row.mid_ticks,
        "microprice_ticks" => row.microprice_ticks,
        "imbalance" => row.imbalance,
        "signed_order_flow" => row.signed_order_flow,
    )
end

_feature_vector(row::FeatureRow) = Float64[
    row.spread_ticks,
    row.mid_ticks,
    row.microprice_ticks,
    row.imbalance,
    row.signed_order_flow,
]

function _forecast_dict(forecast::Forecast, producer::String)
    p_down, p_flat, p_up = forecast.probabilities
    argmax_index = argmax(collect(forecast.probabilities))
    argmax_side = ("down", "flat", "up")[argmax_index]
    return Dict{String, Any}(
        "producer" => producer,
        "horizon_events" => forecast.horizon.events,
        "exchange_ts_ns" => forecast.exchange_ts_ns,
        "sequence" => forecast.sequence,
        "probabilities" => Dict{String, Any}("down" => p_down, "flat" => p_flat, "up" => p_up),
        "argmax" => argmax_side,
        "confidence" => forecast.probabilities[argmax_index],
    )
end

function _market_dict(call::TrajectoryCall)
    snap = call.snapshot
    bid = isempty(snap.bids) ? nothing : first(first(snap.bids))
    ask = isempty(snap.asks) ? nothing : first(first(snap.asks))
    return Dict{String, Any}(
        "venue" => String(snap.venue),
        "instrument" => snap.instrument,
        "exchange_ts_ns" => call.event.exchange_ts_ns,
        "receive_ts_ns" => call.event.receive_ts_ns,
        "sequence" => call.event.sequence,
        "best_bid_ticks" => bid,
        "best_ask_ticks" => ask,
        "spread_ticks" => bid === nothing || ask === nothing ? nothing : ask - bid,
        "mid_ticks" =>
            bid === nothing || ask === nothing ? nothing : Float64(Int128(bid) + Int128(ask)) / 2,
        "features" => call.features === nothing ? nothing : _features_dict(call.features),
    )
end

function _normalize_intents(output)
    output === nothing && return Union{OrderIntent, CancelIntent}[]
    output isa Union{OrderIntent, CancelIntent} && return Union{OrderIntent, CancelIntent}[output]
    output isa AbstractVector || throw(
        ArgumentError(
            "decision function must return nothing, an intent, or a vector of intents, " *
            "got $(typeof(output))",
        ),
    )
    intents = Union{OrderIntent, CancelIntent}[]
    for intent in output
        intent isa Union{OrderIntent, CancelIntent} ||
            throw(ArgumentError("decision vector elements must be intents, got $(typeof(intent))"))
        push!(intents, intent)
    end
    return intents
end

function _intent_dicts(call::TrajectoryCall, records)
    submits = [r for r in records if r.action == :submit]
    cancels = [r for r in records if r.action == :cancel]
    submit_index = 0
    cancel_index = 0
    dicts = Dict{String, Any}[]
    for intent in call.intents
        if intent isa OrderIntent
            submit_index += 1
            record = submits[submit_index]
            push!(
                dicts,
                Dict{String, Any}(
                    "action" => "submit",
                    "order_id" => record.order_id,
                    "side" => lowercase(string(intent.side)),
                    "qty_lots" => intent.qty_lots,
                    "limit_price_ticks" => intent.limit_price_ticks,
                    "time_in_force" => String(intent.time_in_force),
                    "post_only" => intent.post_only,
                    "tag" => intent.tag,
                    "accepted" => record.accepted,
                    "reason" => record.reason,
                    "effective_ts_ns" => record.effective_ts_ns,
                ),
            )
        else
            cancel_index += 1
            record = cancels[cancel_index]
            push!(
                dicts,
                Dict{String, Any}(
                    "action" => "cancel",
                    "order_id" => record.order_id,
                    "accepted" => record.accepted,
                    "reason" => record.reason,
                    "effective_ts_ns" => record.effective_ts_ns,
                ),
            )
        end
    end
    return dicts
end

function _decision_action(call::TrajectoryCall)
    isempty(call.intents) && return "none"
    all(i -> i isa OrderIntent, call.intents) && return "submit"
    all(i -> i isa CancelIntent, call.intents) && return "cancel"
    return "mixed"
end

function _decision_status(call::TrajectoryCall, records, fills)
    intents = _decision_action(call)
    intents == "none" && return "none"
    intents == "cancel" && return "cancel_request"
    submits = [r for r in records if r.action == :submit]
    !isempty(submits) && all(r -> !r.accepted, submits) && return "rejected"
    isempty(fills) || return "executed"
    return "no_fill"
end

# ── Outcome labeling ─────────────────────────────────────────────────────────

# Steps strictly after the decision step at which the horizon resolves; `nothing`
# when the run ends first (recorded as a session-end fallback, never extrapolated).
function _horizon_end_step(ledger, spec::OutcomeHorizonSpec, decision_step::Int)
    if spec.kind == :events
        seen = 0
        for index in (decision_step + 1):length(ledger)
            ledger[index].kind == :venue_event || continue
            seen += 1
            seen == spec.n && return index
        end
        return nothing
    elseif spec.kind == :wall_ns
        cutoff = Base.Checked.checked_add(ledger[decision_step].ts_ns, spec.ns)
        for index in (decision_step + 1):length(ledger)
            ledger[index].ts_ns >= cutoff && return index
        end
        return nothing
    elseif spec.kind == :close
        for index in (decision_step + 1):length(ledger)
            ledger[index].inventory_lots == 0 && return index
        end
        return nothing
    else
        return length(ledger)
    end
end

function _horizon_outcome(
    ledger,
    fills,
    realized_by_fill,
    scale::ParityScale,
    spec::OutcomeHorizonSpec,
    decision_step::Int,
    decision_id::Int64,
)
    resolved = _horizon_end_step(ledger, spec, decision_step)
    fallback = resolved === nothing
    end_index = something(resolved, length(ledger))
    start = ledger[decision_step]
    finish = ledger[end_index]
    pnl =
        start.equity_subunits === nothing || finish.equity_subunits === nothing ? nothing :
        _checked_i64(Int128(finish.equity_subunits) - Int128(start.equity_subunits), "horizon pnl")
    return_ppm =
        pnl === nothing || start.equity_subunits == 0 ? nothing :
        Float64(Int128(pnl) * Int128(1_000_000)) / Float64(start.equity_subunits)
    window_fills =
        [f for f in fills if f.decision_id == decision_id && f.exchange_ts_ns <= finish.ts_ns]
    realized = _checked_i64(
        sum(
            (Int128(get(realized_by_fill, f.fill_id, Int64(0))) for f in window_fills);
            init = Int128(0),
        ),
        "decision realized pnl",
    )
    fees = _checked_i64(
        sum((Int128(f.fee_subunits) for f in window_fills); init = Int128(0)),
        "decision fees",
    )
    net_lots = _checked_i64(
        sum(
            (Int128(f.qty_lots) * (f.side == Buy ? 1 : -1) for f in window_fills);
            init = Int128(0),
        ),
        "decision net lots",
    )
    equities = [
        ledger[index].equity_subunits for
        index in decision_step:end_index if ledger[index].equity_subunits !== nothing
    ]
    mae =
        start.equity_subunits === nothing || length(equities) <= 1 ? nothing :
        _checked_i64(Int128(minimum(equities)) - Int128(start.equity_subunits), "adverse excursion")
    mfe =
        start.equity_subunits === nothing || length(equities) <= 1 ? nothing :
        _checked_i64(
            Int128(maximum(equities)) - Int128(start.equity_subunits),
            "favorable excursion",
        )
    return Dict{String, Any}(
        "name" => spec.name,
        "kind" => String(spec.kind),
        "n" => spec.n,
        "ns" => spec.ns,
        "end_step" => finish.step,
        "end_ts_ns" => finish.ts_ns,
        "end_reason" => fallback ? "session_end_fallback" : "horizon_reached",
        # Equity delta over the whole paper book at both endpoints: it includes the
        # mark-to-market of any pre-existing inventory, not just this decision's fills.
        "paper_pnl_subunits" => pnl,
        "return_ppm" => return_ppm,
        "decision_realized_pnl_subunits" => realized,
        "decision_fees_subunits" => fees,
        "decision_net_lots" => net_lots,
        "decision_fills" => length(window_fills),
        "max_adverse_excursion_subunits" => mae,
        "max_favorable_excursion_subunits" => mfe,
    )
end

function _outcome_dict(
    call,
    records,
    fills_all,
    realized_by_fill,
    ledger,
    scale,
    horizons,
    decision_step,
)
    decision_fills = [f for f in fills_all if f.decision_id == call.decision_id]
    status = _decision_status(call, records, decision_fills)
    if status == "rejected"
        # A rejected decision never traded; attaching equity deltas would fabricate PnL.
        return Dict{String, Any}("status" => status, "horizons" => Dict{String, Any}())
    end
    horizon_outcomes = Dict{String, Any}(
        spec.name => _horizon_outcome(
            ledger,
            fills_all,
            realized_by_fill,
            scale,
            spec,
            decision_step,
            call.decision_id,
        ) for spec in horizons
    )
    return Dict{String, Any}("status" => status, "horizons" => horizon_outcomes)
end

# ── Row emitters ─────────────────────────────────────────────────────────────

function _trajectory_row(
    call,
    phase,
    records,
    fills,
    realized_by_fill,
    ledger,
    scale,
    horizons,
    decision_step,
    run_id,
)
    point = ledger[decision_step]
    return Dict{String, Any}(
        "record_type" => TRAJECTORY_RECORD_TYPE,
        "schema_version" => 1,
        "paper_only" => true,
        "run_id" => run_id,
        "decision_id" => call.decision_id,
        "phase" => String(phase),
        "decision_ts_ns" => call.decision_ts_ns,
        "info_sequence" => call.info_sequence,
        "info_exchange_ts_ns" => call.info_exchange_ts_ns,
        "market" => _market_dict(call),
        "signal" => call.signal,
        "decision" => Dict{String, Any}(
            "action" => _decision_action(call),
            "intents" => _intent_dicts(call, records),
        ),
        "paper_account" => Dict{String, Any}(
            "cash_subunits" => point.cash_subunits,
            "inventory_lots" => point.inventory_lots,
            "total_cost_subunits" => point.total_cost_subunits,
            "realized_pnl_subunits" => point.realized_pnl_subunits,
            "fees_subunits" => point.fees_subunits,
            "mark_price_ticks" => point.mark_price_ticks,
            "mark_status" => String(point.mark_status),
            "equity_subunits" => point.equity_subunits,
            "drawdown_subunits" => point.drawdown_subunits,
            "reported_inventory_lots" => call.reported_inventory_lots,
            "reported_cash_subunits" => call.reported_cash_subunits,
        ),
        "outcome" => _outcome_dict(
            call,
            records,
            fills,
            realized_by_fill,
            ledger,
            scale,
            horizons,
            decision_step,
        ),
    )
end

function _prediction_row(call, phase, records, run_id)
    return Dict{String, Any}(
        "record_type" => PREDICTION_RECORD_TYPE,
        "schema_version" => 1,
        "paper_only" => true,
        "run_id" => run_id,
        "decision_id" => call.decision_id,
        "phase" => String(phase),
        "sequence" => call.info_sequence,
        "decision_ts_ns" => call.decision_ts_ns,
        "exchange_ts_ns" => call.info_exchange_ts_ns,
        "features" => call.features === nothing ? nothing : _features_dict(call.features),
        "forecast" => call.signal,
        "action" => _decision_action(call),
        "accepted" => count(r -> r.accepted, records),
        "rejected" => count(r -> !r.accepted, records),
    )
end

function _snn_decision_target(call, status)
    status == "rejected" && return "rejected"
    submits = [i for i in call.intents if i isa OrderIntent]
    isempty(submits) && return "hold"
    sides = unique(i.side for i in submits)
    length(sides) == 1 || return "mixed"
    return only(sides) == Buy ? "buy" : "sell"
end

function _snn_row(
    call,
    phase,
    records,
    fills,
    realized_by_fill,
    ledger,
    scale,
    horizons,
    decision_step,
    normalizer,
    run_id,
)
    status =
        _decision_status(call, records, [f for f in fills if f.decision_id == call.decision_id])
    features = nothing
    if call.features !== nothing && normalizer.fitted
        frame = FeatureFrame(FeatureRow[call.features])
        transform!(normalizer, frame)
        features = _feature_vector(only(frame))
    end
    outcomes = Dict{String, Any}()
    if status != "rejected"
        for spec in horizons
            outcome = _horizon_outcome(
                ledger,
                fills,
                realized_by_fill,
                scale,
                spec,
                decision_step,
                call.decision_id,
            )
            outcomes[spec.name] = outcome["paper_pnl_subunits"]
        end
    end
    signal = call.signal
    snn_signal =
        signal === nothing ? nothing :
        Dict{String, Any}(
            "confidence" => get(signal, "confidence", nothing),
            "side" => get(signal, "argmax", nothing),
            "probabilities" => get(signal, "probabilities", nothing),
        )
    return Dict{String, Any}(
        "record_type" => SNN_RECORD_TYPE,
        "schema_version" => 1,
        "paper_only" => true,
        "run_id" => run_id,
        "sequence_id" => call.info_sequence,
        "timestamp_ns" => call.decision_ts_ns,
        "phase" => String(phase),
        "features" => features,
        "signal" => snn_signal,
        "decision_target" => _snn_decision_target(call, status),
        "outcome" => outcomes,
    )
end

_checked_i64(value::Integer, what) = _checked_i64(Int128(value), what)
function _checked_i64(value::Int128, what)
    typemin(Int64) <= value <= typemax(Int64) || throw(ArgumentError("$what overflows Int64"))
    return Int64(value)
end
