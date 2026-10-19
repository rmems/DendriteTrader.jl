# SPDX-License-Identifier: MIT OR Apache-2.0

function _metrics(state)
    isempty(state.ledger) &&
        throw(ArgumentError("_metrics requires a non-empty ledger (call after session_end)"))
    orders = state.orders
    fills = state.fills
    accepted = [o for o in orders if o.status != :rejected]
    accepted_lots = sum((o.intent.qty_lots for o in accepted); init = Int64(0))
    filled_lots = sum((f.qty_lots for f in fills); init = Int64(0))
    final = last(state.ledger)
    p = state.portfolio
    return SimulationMetrics(
        state.decision_counter,
        length(orders),
        length(accepted),
        count(o -> o.status == :rejected, orders),
        count(o -> o.status == :cancelled, orders),
        count(o -> o.status == :filled, orders),
        count(o -> o.status == :expired, orders),
        length(fills),
        count(f -> f.liquidity == :maker, fills),
        count(f -> f.liquidity == :taker, fills),
        filled_lots,
        accepted_lots,
        accepted_lots == 0 ? 0.0 : filled_lots / accepted_lots,
        _checked_i64(
            sum(
                (Int128(notional_subunits(p.scale, f.price_ticks, f.qty_lots)) for f in fills);
                init = Int128(0),
            ),
            "turnover",
        ),
        p.fees_subunits,
        _checked_i64(
            sum((Int128(f.slippage_subunits) for f in fills); init = Int128(0)),
            "slippage",
        ),
        p.realized_pnl_subunits,
        p.inventory_lots,
        final.mark_price_ticks,
        final.mark_status,
        final.unrealized_pnl_subunits,
        final.equity_subunits,
        maximum((point.drawdown_subunits for point in state.ledger); init = Int64(0)),
        count(point -> point.mark_status == :stale, state.ledger),
        count(point -> point.mark_status == :missing, state.ledger),
    )
end

# ── Normalized records (decision/fill halves of a later trajectory export) ───

_sym(x::Symbol) = String(x)
_sym(::Nothing) = nothing
_side(x::TradeSide) = _side_string(x)
_side(::Nothing) = nothing

"""
    decision_records(result) -> Vector{Dict{String,Any}}

JSON-ready normalized decision records with `record_type = "paper_decision"`. They
carry information, decision and activation/effective times separately and never an
outcome. Intended for the paper-trading trajectory export (LIM-1035 / #76), which is
not implemented here.
"""
decision_records(result::SimulationResult) = [
    Dict{String, Any}(
        "record_type" => "paper_decision",
        "schema_version" => 1,
        "paper_only" => true,
        "instrument" => result.instrument,
        "decision_id" => d.decision_id,
        "decision_ts_ns" => d.decision_ts_ns,
        "info_available_ts_ns" => d.info_available_ts_ns,
        "info_sequence" => d.info_sequence,
        "action" => String(d.action),
        "order_id" => d.order_id,
        "side" => _side(d.side),
        "qty_lots" => d.qty_lots,
        "limit_price_ticks" => d.limit_price_ticks,
        "time_in_force" => _sym(d.time_in_force),
        "post_only" => d.post_only,
        "tag" => d.tag,
        "accepted" => d.accepted,
        "reason" => d.reason,
        "effective_ts_ns" => d.effective_ts_ns,
        "reported_inventory_lots" => d.reported_inventory_lots,
    ) for d in result.decisions
]

"""
    fill_records(result) -> Vector{Dict{String,Any}}

JSON-ready normalized fill records with `record_type = "paper_fill"` and
`approximation = true` (L2 queue inference). Fees and slippage are in quote subunits.
"""
fill_records(result::SimulationResult) = [
    Dict{String, Any}(
        "record_type" => "paper_fill",
        "schema_version" => 1,
        "paper_only" => true,
        "approximation" => true,
        "instrument" => result.instrument,
        "fill_id" => f.fill_id,
        "order_id" => f.order_id,
        "decision_id" => f.decision_id,
        "side" => _side(f.side),
        "price_ticks" => f.price_ticks,
        "qty_lots" => f.qty_lots,
        "liquidity" => String(f.liquidity),
        "fee_subunits" => f.fee_subunits,
        "slippage_subunits" => f.slippage_subunits,
        "exchange_ts_ns" => f.exchange_ts_ns,
        "report_ts_ns" => f.report_ts_ns,
        "activation_ts_ns" => f.activation_ts_ns,
        "decision_ts_ns" => f.decision_ts_ns,
        "info_available_ts_ns" => f.info_available_ts_ns,
        "info_sequence" => f.info_sequence,
        "trigger" => String(f.trigger),
        "trigger_sequence" => f.trigger_sequence,
    ) for f in result.fills
]

_order_dict(o::PaperOrder) = Dict{String, Any}(
    "order_id" => o.order_id,
    "decision_id" => o.decision_id,
    "side" => _side(o.intent.side),
    "qty_lots" => o.intent.qty_lots,
    "limit_price_ticks" => o.intent.limit_price_ticks,
    "time_in_force" => String(o.intent.time_in_force),
    "post_only" => o.intent.post_only,
    "tag" => o.intent.tag,
    "info_sequence" => o.info_sequence,
    "info_available_ts_ns" => o.info_available_ts_ns,
    "decision_ts_ns" => o.decision_ts_ns,
    "activation_ts_ns" => o.activation_ts_ns,
    "status" => String(o.status),
    "remaining_lots" => o.remaining_lots,
    "filled_lots" => o.filled_lots,
    "queue_ahead_lots" => o.queue_ahead_lots,
    "initial_queue_ahead_lots" => o.initial_queue_ahead_lots,
    "join_seq" => o.join_seq,
    "terminal_ts_ns" => o.terminal_ts_ns,
    "terminal_reason" => o.terminal_reason,
)

_ledger_dict(point::LedgerPoint) = Dict{String, Any}(
    String(name) => (v = getfield(point, name); v isa Symbol ? String(v) : v) for
    name in fieldnames(LedgerPoint)
)

_metrics_dict(m::SimulationMetrics) = Dict{String, Any}(
    String(name) => (v = getfield(m, name); v isa Symbol ? String(v) : v) for
    name in fieldnames(SimulationMetrics)
)

# Canonical JSON: sorted object keys, no whitespace. Floats use Julia's shortest repr.
function _canon(io::IO, x::AbstractDict)
    print(io, '{')
    for (index, key) in enumerate(sort!(collect(keys(x)); by = string))
        index > 1 && print(io, ',')
        JSON.print(io, string(key))
        print(io, ':')
        _canon(io, x[key])
    end
    print(io, '}')
end
function _canon(io::IO, x::AbstractVector)
    print(io, '[')
    for (index, value) in enumerate(x)
        index > 1 && print(io, ',')
        _canon(io, value)
    end
    print(io, ']')
end
_canon(io::IO, x::Union{Nothing, Bool, Integer, AbstractString}) = JSON.print(io, x)
_canon(io::IO, x::AbstractFloat) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "null")
_canon(io::IO, x::Symbol) = JSON.print(io, String(x))

"""Canonical (key-sorted, compact) JSON string of a JSON-like value."""
canonical_json(x) = sprint(_canon, x)

function _canonical_payload(
    assumptions,
    decisions,
    orders,
    fills,
    ledger,
    audit,
    metrics,
    instrument::AbstractString,
)
    shell = SimulationResult(
        assumptions,
        decisions,
        orders,
        fills,
        ledger,
        audit,
        metrics,
        false,
        0,
        0,
        0,
        String(instrument),
        ParityScale(),
        "",
    )
    return canonical_json(
        Dict{String, Any}(
            "assumptions" => assumptions,
            "decisions" => decision_records(shell),
            "orders" => [_order_dict(o) for o in orders],
            "fills" => fill_records(shell),
            "ledger" => [_ledger_dict(point) for point in ledger],
            "audit" => [audit_dict(row) for row in audit],
            "metrics" => _metrics_dict(metrics),
        ),
    )
end

"""
    reconciliation_errors(result) -> Vector{String}

Independently re-derive cash, signed inventory, fees, the accounting identity,
unrealized PnL, equity, peak and drawdown at every ledger step from the fill list and
the recorded marks; return every discrepancy (empty when the run reconciles exactly).
"""
function reconciliation_errors(result::SimulationResult)
    errors = String[]
    scale = result.scale
    cash = Int128(result.initial_cash_subunits)
    inventory = Int128(result.initial_inventory_lots)
    fees = Int128(0)
    peak = Int128(result.initial_cash_subunits) + result.initial_total_cost_subunits
    applied = 0
    for point in result.ledger
        point.fills_applied >= applied || push!(errors, "step $(point.step): fill count decreased")
        point.fills_applied <= length(result.fills) || push!(
            errors,
            "step $(point.step): fills_applied $(point.fills_applied) exceeds $(length(result.fills)) fills",
        )
        while applied < min(point.fills_applied, length(result.fills))
            applied += 1
            fill = result.fills[applied]
            fill.exchange_ts_ns <= point.ts_ns ||
                push!(errors, "fill $(fill.fill_id) is booked before its venue time")
            notional = Int128(fill.price_ticks) * fill.qty_lots ÷ scale.qty_scale
            cash +=
                fill.side == Buy ? -(notional + fill.fee_subunits) : notional - fill.fee_subunits
            inventory += fill.side == Buy ? fill.qty_lots : -fill.qty_lots
            fees += fill.fee_subunits
        end
        point.cash_subunits == cash ||
            push!(errors, "step $(point.step): cash $(point.cash_subunits) != $cash")
        point.inventory_lots == inventory ||
            push!(errors, "step $(point.step): inventory $(point.inventory_lots) != $inventory")
        point.fees_subunits == fees ||
            push!(errors, "step $(point.step): fees $(point.fees_subunits) != $fees")
        Int128(point.cash_subunits) + point.total_cost_subunits ==
        Int128(result.initial_cash_subunits) +
        result.initial_total_cost_subunits +
        point.realized_pnl_subunits ||
            push!(errors, "step $(point.step): cash + cost != initial book value + realized PnL")
        if point.inventory_lots == 0
            point.unrealized_pnl_subunits == 0 ||
                push!(errors, "step $(point.step): flat book has unrealized PnL")
            point.total_cost_subunits == 0 ||
                push!(errors, "step $(point.step): flat book retains cost")
        elseif point.mark_price_ticks === nothing
            point.unrealized_pnl_subunits === nothing ||
                push!(errors, "step $(point.step): unrealized PnL without a mark")
            point.equity_subunits === nothing ||
                push!(errors, "step $(point.step): equity without a mark")
            # The simulator leaves peak unchanged and reports zero drawdown when equity is
            # unknown; check that instead of skipping the step.
            point.peak_equity_subunits == peak ||
                push!(errors, "step $(point.step): peak changed without a mark")
            point.drawdown_subunits == 0 ||
                push!(errors, "step $(point.step): drawdown without a mark")
            continue
        else
            mark_value = Int128(point.mark_price_ticks) * point.inventory_lots ÷ scale.qty_scale
            expected = mark_value - point.total_cost_subunits
            point.unrealized_pnl_subunits == expected || push!(
                errors,
                "step $(point.step): unrealized $(point.unrealized_pnl_subunits) != $expected",
            )
        end
        equity =
            Int128(point.cash_subunits) + point.total_cost_subunits + point.unrealized_pnl_subunits
        point.equity_subunits == equity ||
            push!(errors, "step $(point.step): equity $(point.equity_subunits) != $equity")
        peak = max(peak, equity)
        point.peak_equity_subunits == peak ||
            push!(errors, "step $(point.step): peak $(point.peak_equity_subunits) != $peak")
        point.drawdown_subunits == peak - equity || push!(
            errors,
            "step $(point.step): drawdown $(point.drawdown_subunits) != $(peak - equity)",
        )
    end
    applied == length(result.fills) ||
        push!(errors, "$(length(result.fills) - applied) fills never reached the ledger")
    return errors
end
