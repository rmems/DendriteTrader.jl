# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    Simulation

Latency-aware, event-driven **paper** execution simulator for research (RM-1864).

It replays a canonical `ReplaySession`, calls a decision function only with
information observable under a declared feed-delay model, activates hypothetical
limit orders after a declared fixed latency, infers fills with a disclosed
`L2QueueApproximation`, and reconciles an integer `PaperPortfolio` after every event.

Nothing here submits orders, holds credentials, or talks to a network. Fills are
hypothetical inferences from L2 data, not observed executions, and imply no venue
matching-engine or market-impact realism. `PaperPortfolio` is an evaluation state,
not an authoritative ledger; persistent accounting belongs to `metabolic-ledger`.
"""
module Simulation

using JSON
using SHA
using ..DendriteTrader: TradeSide, Buy, Sell, Neutral
using ..DendriteTrader.Market:
    MarketEvent,
    BookDelta,
    TradePrint,
    BookSnapshot,
    BookSide,
    Bid,
    Ask,
    OrderBookState,
    ReplaySession,
    apply!,
    snapshot,
    best_bid,
    best_ask
using ..DendriteTrader.Features: FeatureRow, FeatureFrame, MovementTarget, microstructure_features
using ..DendriteTrader.Models: AbstractForecastModel, ForecastHorizon, Forecast, predict!

export OrderIntent, CancelIntent, RiskLimits, PaperOrder, OpenOrderView
export FixedLatency, AbstractFeedDelay, ReceiveTimestampFeed, FixedFeedDelay
export activation_time, cancel_effective_time, available_time
export FeeSchedule, fill_fee, L2QueueApproximation, Fill
export ParityScale, PortfolioError, PaperPortfolio, AuditRow, apply_fill!, record_non_fill!
export notional_subunits, unrealized_pnl, weighted_cost_basis_ticks, in_parity_subset, reconciles
export SimulationConfig, DecisionContext, DecisionRecord, LedgerPoint, SimulationResult
export SimulationMetrics, simulate!, ForecastPolicy, reconciliation_errors
export decision_records, fill_records, audit_dict
export PARITY_SCHEMA_ID, load_parity_fixture, run_parity_fixture, parity_document

include("Orders.jl")
include("Latency.jl")
include("Portfolio.jl")
include("Fills.jl")

"""
    SimulationConfig(; latency, feed=ReceiveTimestampFeed(), fill_model=L2QueueApproximation(),
                       fees=FeeSchedule(), limits=RiskLimits())

Every execution assumption of a run. All of it is copied into
`SimulationResult.assumptions` and into the result hash.
"""
struct SimulationConfig
    latency::FixedLatency
    feed::AbstractFeedDelay
    fill_model::L2QueueApproximation
    fees::FeeSchedule
    limits::RiskLimits
end

SimulationConfig(;
    latency::FixedLatency,
    feed::AbstractFeedDelay = ReceiveTimestampFeed(),
    fill_model::L2QueueApproximation = L2QueueApproximation(),
    fees::FeeSchedule = FeeSchedule(),
    limits::RiskLimits = RiskLimits(),
) = SimulationConfig(latency, feed, fill_model, fees, limits)

# ── Decision path ────────────────────────────────────────────────────────────

"""
    DecisionContext

Everything a decision function may read. It is built only from events observable at
`decision_ts_ns` under the feed-delay model and from fills already *reported*
(`fill time + report_ns <= decision_ts_ns`). It carries no label, target or outcome;
reading any other property throws. `snapshot` is a private copy of the observed book.
`features` is the causal `microstructure_features` row for the observed snapshot, or
`nothing` while the observed book is one-sided.
"""
struct DecisionContext
    decision_ts_ns::Int64
    info_sequence::Int64
    info_exchange_ts_ns::Int64
    event::MarketEvent
    snapshot::BookSnapshot
    features::Union{Nothing, FeatureRow}
    reported_inventory_lots::Int64
    reported_cash_subunits::Int64
    open_orders::Vector{OpenOrderView}
end

const _CONTEXT_FIELDS = fieldnames(DecisionContext)

function Base.getproperty(ctx::DecisionContext, name::Symbol)
    name in _CONTEXT_FIELDS || throw(
        ArgumentError(
            "DecisionContext has no property $(name): labels, targets and outcomes are " *
            "not available on the decision path",
        ),
    )
    return getfield(ctx, name)
end

_is_label_type(T) = T <: MovementTarget || (T <: AbstractArray && eltype(T) <: MovementTarget)

function _scan_label_capture(x, depth::Int)
    x isa MovementTarget && return true
    _is_label_type(typeof(x)) && return true
    depth <= 0 && return false
    x isa Union{Number, AbstractString, Symbol, Nothing, Function, Module, Type} &&
        !(x isa Function && nfields(x) > 0) &&
        return false
    if x isa Core.Box
        return isdefined(x, :contents) && _scan_label_capture(x.contents, depth - 1)
    elseif x isa AbstractDict
        return any(value -> _scan_label_capture(value, depth - 1), values(x))
    elseif x isa Union{AbstractArray, Tuple, NamedTuple}
        isbitstype(eltype(x)) && return false
        for index in eachindex(x)
            x isa AbstractArray && !isassigned(x, index) && continue
            _scan_label_capture(x[index], depth - 1) && return true
        end
        return false
    end
    isstructtype(typeof(x)) || return false
    for index in 1:nfields(x)
        isdefined(x, index) || continue
        _scan_label_capture(getfield(x, index), depth - 1) && return true
    end
    return false
end

"""
Runtime guard: reject a decision function (closure or callable struct) that captures
`MovementTarget` labels. This is a guard against accidental capture, not a sandbox;
the structural guarantee is that `DecisionContext` exposes no label/outcome data.
"""
function _assert_label_free(decide)
    _scan_label_capture(decide, 4) && throw(
        ArgumentError(
            "decision function captures MovementTarget labels; future outcomes must not " *
            "be reachable from the decision path",
        ),
    )
    return nothing
end

"""
    DecisionRecord

Normalized record of one submit or cancel request (accepted or rejected), with the
information, decision and activation clocks kept separate. Intended as the decision
half of a later paper-trading trajectory export (LIM-1035 / DendriteTrader #76).
"""
struct DecisionRecord
    decision_id::Int64
    decision_ts_ns::Int64
    info_available_ts_ns::Int64
    info_sequence::Int64
    action::Symbol
    order_id::Int64
    side::Union{Nothing, TradeSide}
    qty_lots::Union{Nothing, Int64}
    limit_price_ticks::Union{Nothing, Int64}
    time_in_force::Union{Nothing, Symbol}
    post_only::Bool
    tag::String
    accepted::Bool
    reason::Union{Nothing, String}
    effective_ts_ns::Union{Nothing, Int64}
    reported_inventory_lots::Int64
end

"""
    LedgerPoint

Per-step evaluation state. `mark_price_ticks` is derived only from the venue book at or
before `ts_ns` (mid rounded half a tick against the position). `mark_status` is
`:fresh`, `:stale` (book one-sided; last valid mid reused) or `:missing` (no valid mid
yet). When a non-flat position has no mark, `unrealized_pnl_subunits` and
`equity_subunits` are `nothing` rather than a guessed value.
"""
struct LedgerPoint
    step::Int64
    ts_ns::Int64
    kind::Symbol
    ref::Int64
    cash_subunits::Int64
    inventory_lots::Int64
    total_cost_subunits::Int64
    realized_pnl_subunits::Int64
    fees_subunits::Int64
    mark_price_ticks::Union{Nothing, Int64}
    mark_status::Symbol
    unrealized_pnl_subunits::Union{Nothing, Int64}
    equity_subunits::Union{Nothing, Int64}
    peak_equity_subunits::Int64
    drawdown_subunits::Int64
    fills_applied::Int
end

"""
    SimulationMetrics

Integer summary of one paper run. `final_equity_subunits` values any open position at
the last valid venue mid (terminal valuation); it is `nothing` if a position is open
and no valid mid was ever observed. `max_drawdown_subunits` is peak-to-trough over
marked steps. `fill_rate` is filled lots over lots of accepted orders.
"""
struct SimulationMetrics
    decisions::Int
    orders_submitted::Int
    orders_accepted::Int
    orders_rejected::Int
    orders_cancelled::Int
    orders_filled::Int
    orders_expired::Int
    fills::Int
    maker_fills::Int
    taker_fills::Int
    filled_lots::Int64
    accepted_lots::Int64
    fill_rate::Float64
    turnover_subunits::Int64
    fees_subunits::Int64
    slippage_subunits::Int64
    realized_pnl_subunits::Int64
    final_inventory_lots::Int64
    final_mark_price_ticks::Union{Nothing, Int64}
    final_mark_status::Symbol
    final_unrealized_pnl_subunits::Union{Nothing, Int64}
    final_equity_subunits::Union{Nothing, Int64}
    max_drawdown_subunits::Int64
    stale_mark_steps::Int
    missing_mark_steps::Int
end

"""
    SimulationResult

Output of `simulate!`: declared `assumptions` (latency, feed, fill model, seed, fees,
limits, accounting mode), normalized decision and fill records, final order states,
the per-step `ledger`, the contract-shaped `audit`, summary `metrics`, whether the run
stayed in the long/flat parity subset, and a SHA-256 `result_hash` of the canonical
JSON of all of the above.
"""
struct SimulationResult
    assumptions::Dict{String, Any}
    decisions::Vector{DecisionRecord}
    orders::Vector{PaperOrder}
    fills::Vector{Fill}
    ledger::Vector{LedgerPoint}
    audit::Vector{AuditRow}
    metrics::SimulationMetrics
    in_parity_subset::Bool
    initial_cash_subunits::Int64
    initial_inventory_lots::Int64
    initial_total_cost_subunits::Int64
    instrument::String
    scale::ParityScale
    result_hash::String
end

include("Metrics.jl")
include("Parity.jl")

# ── Simulator state ──────────────────────────────────────────────────────────

const _RANK_VENUE = 0
const _RANK_ACTIVATION = 1
const _RANK_CANCEL = 2
const _RANK_OBSERVATION = 3

struct _Action
    ts_ns::Int64
    rank::Int
    id::Int64
    order_id::Int64
end

_key(a::_Action) = (a.ts_ns, a.rank, a.id)

mutable struct _SimState
    config::SimulationConfig
    portfolio::PaperPortfolio
    start_cash::Int64
    start_inventory::Int64
    venue::OrderBookState
    observed::OrderBookState
    depletion::Dict{Tuple{BookSide, Int64}, Int64}
    orders::Vector{PaperOrder}
    fills::Vector{Fill}
    fill_cash_flows::Vector{Int64}
    fills_by_order::Dict{Int64, Vector{Int}}
    actions::Vector{_Action}
    action_counter::Int64
    decisions::Vector{DecisionRecord}
    ledger::Vector{LedgerPoint}
    last_mark_twice::Union{Nothing, Int64}
    peak::Int64
    step::Int64
    decision_counter::Int64
    last_complete_observed::Union{Nothing, BookSnapshot}
end

function _schedule!(state::_SimState, ts::Int64, rank::Int, order_id::Int64)
    state.action_counter += 1
    action = _Action(ts, rank, state.action_counter, order_id)
    index = searchsortedfirst(state.actions, action; by = _key)
    insert!(state.actions, index, action)
    return action
end

_live_orders(state::_SimState) = (o for o in state.orders if _is_live(o))

function _splitmix64(x::UInt64)
    z = x + 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# ── Cash reservation and risk checks (reject before mutation) ────────────────

function _buy_reserve(state::_SimState, limit::Int64, qty::Int64)
    scale = state.portfolio.scale
    notional = notional_subunits(scale, limit, qty)
    fees = state.config.fees
    slack = Int128(qty) * (fees.per_fill_fee_subunits + 1)
    return _checked_i64(Int128(notional) + _max_fee(fees, notional) + slack, "cash reserve")
end

function _risk_reject_reason(state::_SimState, intent::OrderIntent)
    limits = state.config.limits
    p = state.portfolio
    intent.qty_lots <= limits.max_order_lots || return "order_size_limit"
    count(_ -> true, _live_orders(state)) < limits.max_open_orders || return "open_order_limit"
    live_buy =
        sum((o.remaining_lots for o in _live_orders(state) if o.intent.side == Buy); init = 0)
    live_sell =
        sum((o.remaining_lots for o in _live_orders(state) if o.intent.side == Sell); init = 0)
    if intent.side == Buy
        Int128(p.inventory_lots) + live_buy + intent.qty_lots <= limits.max_inventory_lots ||
            return "inventory_limit"
        reserved = sum(
            (
                Int128(_buy_reserve(state, o.intent.limit_price_ticks, o.remaining_lots)) for
                o in _live_orders(state) if o.intent.side == Buy
            );
            init = Int128(0),
        )
        Int128(_buy_reserve(state, intent.limit_price_ticks, intent.qty_lots)) <=
        Int128(p.cash_subunits) - reserved || return "insufficient_cash"
    else
        worst = Int128(p.inventory_lots) - live_sell - intent.qty_lots
        if p.mode == :long_flat
            worst >= 0 || return "unsupported_short"
        else
            worst >= -Int128(limits.max_inventory_lots) || return "inventory_limit"
        end
    end
    return nothing
end

# ── Fills ────────────────────────────────────────────────────────────────────

function _record_fill!(
    state::_SimState,
    order::PaperOrder,
    price::Int64,
    reference_price::Int64,
    qty::Int64,
    liquidity::Symbol,
    ts::Int64,
    trigger::Symbol,
    trigger_sequence::Int64,
)
    p = state.portfolio
    side = order.intent.side
    ts >= order.activation_ts_ns || error("internal: fill before activation")
    notional = notional_subunits(p.scale, price, qty)
    reference_notional = notional_subunits(p.scale, reference_price, qty)
    slippage = side == Buy ? notional - reference_notional : reference_notional - notional
    fee = fill_fee(state.config.fees, notional, liquidity)
    fill_number = length(get(state.fills_by_order, order.order_id, Int[])) + 1
    fill_id = "$(_order_ref(order.order_id))-f$(fill_number)"
    apply_fill!(p, fill_id, ts, p.instrument, side, price, qty, fee)
    fill = Fill(
        fill_id,
        order.order_id,
        order.decision_id,
        side,
        price,
        qty,
        liquidity,
        fee,
        slippage,
        ts,
        _checked_time(ts, state.config.latency.report_ns),
        order.activation_ts_ns,
        order.decision_ts_ns,
        order.info_available_ts_ns,
        order.info_sequence,
        trigger,
        trigger_sequence,
    )
    push!(state.fills, fill)
    push!(state.fill_cash_flows, side == Buy ? -(notional + fee) : notional - fee)
    push!(get!(state.fills_by_order, order.order_id, Int[]), length(state.fills))
    order.remaining_lots -= qty
    order.filled_lots += qty
    if order.remaining_lots == 0
        order.status = :filled
        order.resting = false
        order.terminal_ts_ns = ts
        order.terminal_reason = "filled"
    end
    return fill
end

function _terminate!(state::_SimState, order::PaperOrder, status::Symbol, kind::Symbol, ts, reason)
    order.status = status
    order.resting = false
    order.terminal_ts_ns = ts
    order.terminal_reason = reason
    record_non_fill!(
        state.portfolio,
        kind,
        _order_ref(order.order_id),
        ts,
        state.portfolio.instrument,
        reason,
    )
    return order
end

function _activate!(state::_SimState, order::PaperOrder, ts::Int64)
    order.status == :pending || return nothing
    order.status = :active
    model = state.config.fill_model
    intent = order.intent
    book = state.venue
    if _is_marketable(book, state.depletion, model, intent)
        if intent.post_only
            _terminate!(state, order, :rejected, :reject, ts, "post_only_would_take")
            return nothing
        end
        opposite = _opposite_book_side(intent.side)
        for price in _opposite_levels(book, intent.side)
            order.remaining_lots == 0 && break
            fill_price = _slipped(intent.side, price, model.taker_slippage_ticks)
            _within_limit(intent.side, fill_price, intent.limit_price_ticks) || break
            available = _effective_size(book, state.depletion, opposite, price)
            available == 0 && continue
            qty = min(available, order.remaining_lots)
            _deplete!(state.depletion, opposite, price, qty)
            _record_fill!(
                state,
                order,
                fill_price,
                price,
                qty,
                :taker,
                ts,
                :activation_sweep,
                book.sequence,
            )
        end
    end
    order.remaining_lots == 0 && return nothing
    if intent.time_in_force == :ioc
        _terminate!(state, order, :cancelled, :cancel, ts, "ioc_remainder")
        return nothing
    end
    own = _own_book_side(intent.side)
    displayed = _effective_size(book, state.depletion, own, intent.limit_price_ticks)
    ahead = if model.queue_position == :back
        displayed
    else
        Int64(_splitmix64(model.seed ⊻ UInt64(order.order_id)) % UInt64(displayed + 1))
    end
    own_ahead = sum(
        (
            o.remaining_lots for o in state.orders if o.resting &&
                _is_live(o) &&
                o.intent.side == intent.side &&
                o.intent.limit_price_ticks == intent.limit_price_ticks
        );
        init = Int64(0),
    )
    order.queue_ahead_lots = ahead + own_ahead
    order.initial_queue_ahead_lots = order.queue_ahead_lots
    order.resting = true
    return nothing
end

function _resting_orders(state::_SimState)
    return [o for o in state.orders if o.resting && o.status == :active]
end

function _on_book_delta!(state::_SimState, event::BookDelta)
    levels = _levels(state.venue, event.side)
    displayed = get(levels, event.price_ticks, Int64(0))
    displayed == 0 && delete!(state.depletion, (event.side, event.price_ticks))
    for order in _resting_orders(state)
        intent = order.intent
        if event.side == _own_book_side(intent.side) &&
           event.price_ticks == intent.limit_price_ticks
            event.size_delta < 0 &&
                (order.queue_ahead_lots = min(order.queue_ahead_lots, displayed))
        elseif event.size_delta > 0 &&
               event.side == _opposite_book_side(intent.side) &&
               _strictly_through(intent.side, event.price_ticks, intent.limit_price_ticks)
            available = _effective_size(state.venue, state.depletion, event.side, event.price_ticks)
            qty = min(available, order.remaining_lots)
            qty > 0 || continue
            _deplete!(state.depletion, event.side, event.price_ticks, qty)
            _record_fill!(
                state,
                order,
                intent.limit_price_ticks,
                intent.limit_price_ticks,
                qty,
                :maker,
                event.exchange_ts_ns,
                :quote_through,
                event.sequence,
            )
        end
    end
    return nothing
end

function _on_trade_print!(state::_SimState, event::TradePrint)
    # Each order's queue is advanced by the full print size (everything that traded at
    # its level was ahead of it, including earlier simulated orders it queued behind).
    # Simulated fills from one print never exceed the print size in total.
    leftover_print = event.size
    for order in _resting_orders(state)
        intent = order.intent
        hits_order = intent.side == Buy ? event.aggressor_side == Sell : event.aggressor_side == Buy
        hits_order || continue
        _within_limit(intent.side, event.price_ticks, intent.limit_price_ticks) || continue
        # A print strictly through the limit implies the whole queue at the limit traded.
        _strictly_through(intent.side, event.price_ticks, intent.limit_price_ticks) &&
            (order.queue_ahead_lots = 0)
        ahead = order.queue_ahead_lots
        order.queue_ahead_lots = max(Int64(0), ahead - event.size)
        qty = min(order.remaining_lots, max(Int64(0), event.size - ahead), leftover_print)
        qty > 0 || continue
        leftover_print -= qty
        _record_fill!(
            state,
            order,
            intent.limit_price_ticks,
            intent.limit_price_ticks,
            qty,
            :maker,
            event.exchange_ts_ns,
            :trade_print,
            event.sequence,
        )
    end
    return nothing
end

# ── Observation / decisions ──────────────────────────────────────────────────

function _reported_view(state::_SimState, ts::Int64)
    report_ns = state.config.latency.report_ns
    inventory = state.start_inventory
    cash = state.start_cash
    for (index, fill) in enumerate(state.fills)
        fill.report_ts_ns <= ts || continue
        inventory += fill.side == Buy ? fill.qty_lots : -fill.qty_lots
        cash += state.fill_cash_flows[index]
    end
    views = OpenOrderView[]
    for order in state.orders
        order.status == :rejected && continue
        if order.terminal_ts_ns !== nothing && _checked_time(order.terminal_ts_ns, report_ns) <= ts
            continue
        end
        reported = sum(
            (
                state.fills[i].qty_lots for
                i in get(state.fills_by_order, order.order_id, Int[]) if
                state.fills[i].report_ts_ns <= ts
            );
            init = Int64(0),
        )
        push!(
            views,
            OpenOrderView(
                order.order_id,
                order.intent.side,
                order.intent.limit_price_ticks,
                order.intent.qty_lots,
                reported,
            ),
        )
    end
    return inventory, cash, views
end

function _normalize_intents(output)
    output === nothing && return Union{OrderIntent, CancelIntent}[]
    output isa Union{OrderIntent, CancelIntent} && return Union{OrderIntent, CancelIntent}[output]
    output isa AbstractVector || throw(
        ArgumentError("decision function must return nothing, an intent, or a vector of intents"),
    )
    intents = Union{OrderIntent, CancelIntent}[]
    for item in output
        item isa Union{OrderIntent, CancelIntent} ||
            throw(ArgumentError("decision output contains a non-intent value"))
        push!(intents, item)
    end
    return intents
end

function _submit!(state::_SimState, ctx::DecisionContext, decision_id::Int64, intent::OrderIntent)
    order_id = Int64(length(state.orders) + 1)
    reason = _risk_reject_reason(state, intent)
    decision_ts = ctx.decision_ts_ns
    activation = reason === nothing ? activation_time(state.config.latency, decision_ts) : nothing
    order = PaperOrder(
        order_id,
        decision_id,
        intent,
        ctx.info_sequence,
        decision_ts,
        decision_ts,
        something(activation, decision_ts),
        reason === nothing ? :pending : :rejected,
        intent.qty_lots,
        0,
        0,
        0,
        false,
        nothing,
        nothing,
    )
    push!(state.orders, order)
    if reason === nothing
        _schedule!(state, activation, _RANK_ACTIVATION, order_id)
    else
        order.terminal_ts_ns = decision_ts
        order.terminal_reason = reason
        record_non_fill!(
            state.portfolio,
            :reject,
            _order_ref(order_id),
            decision_ts,
            state.portfolio.instrument,
            reason,
        )
    end
    push!(
        state.decisions,
        DecisionRecord(
            decision_id,
            decision_ts,
            decision_ts,
            ctx.info_sequence,
            :submit,
            order_id,
            intent.side,
            intent.qty_lots,
            intent.limit_price_ticks,
            intent.time_in_force,
            intent.post_only,
            intent.tag,
            reason === nothing,
            reason,
            activation,
            ctx.reported_inventory_lots,
        ),
    )
    return order
end

function _request_cancel!(
    state::_SimState,
    ctx::DecisionContext,
    decision_id::Int64,
    intent::CancelIntent,
)
    decision_ts = ctx.decision_ts_ns
    valid = 1 <= intent.order_id <= length(state.orders) && _is_live(state.orders[intent.order_id])
    effective = valid ? cancel_effective_time(state.config.latency, decision_ts) : nothing
    valid && _schedule!(state, effective, _RANK_CANCEL, intent.order_id)
    if !valid
        record_non_fill!(
            state.portfolio,
            :reject,
            _order_ref(intent.order_id),
            decision_ts,
            state.portfolio.instrument,
            "cancel_unknown_or_terminal",
        )
    end
    push!(
        state.decisions,
        DecisionRecord(
            decision_id,
            decision_ts,
            decision_ts,
            ctx.info_sequence,
            :cancel,
            intent.order_id,
            nothing,
            nothing,
            nothing,
            nothing,
            false,
            "",
            valid,
            valid ? nothing : "cancel_unknown_or_terminal",
            effective,
            ctx.reported_inventory_lots,
        ),
    )
    return nothing
end

function _observe!(state::_SimState, session::ReplaySession, event::MarketEvent, ts::Int64, decide)
    apply!(state.observed, event; policy = session.policy)
    snap = snapshot(state.observed)
    features = nothing
    if !isempty(snap.bids) && !isempty(snap.asks)
        history =
            state.last_complete_observed === nothing ? [snap] : [state.last_complete_observed, snap]
        features = last(microstructure_features(history; allow_sequence_gaps = true))
        state.last_complete_observed = snap
    end
    inventory, cash, views = _reported_view(state, ts)
    ctx = DecisionContext(
        ts,
        event.sequence,
        event.exchange_ts_ns,
        event,
        snap,
        features,
        inventory,
        cash,
        views,
    )
    state.decision_counter += 1
    decision_id = state.decision_counter
    for intent in _normalize_intents(decide(ctx))
        if intent isa OrderIntent
            _submit!(state, ctx, decision_id, intent)
        else
            _request_cancel!(state, ctx, decision_id, intent)
        end
    end
    return nothing
end

# ── Mark-to-market ───────────────────────────────────────────────────────────

function _mark_point!(state::_SimState, ts::Int64, kind::Symbol, ref::Int64)
    p = state.portfolio
    bid = best_bid(state.venue)
    ask = best_ask(state.venue)
    status = if bid !== nothing && ask !== nothing
        state.last_mark_twice = first(bid) + first(ask)
        :fresh
    elseif state.last_mark_twice !== nothing
        :stale
    else
        :missing
    end
    mark = if state.last_mark_twice === nothing
        nothing
    else
        # Round a half-tick mid against the position: down for longs, up for shorts.
        p.inventory_lots >= 0 ? fld(state.last_mark_twice, 2) : cld(state.last_mark_twice, 2)
    end
    unrealized =
        p.inventory_lots == 0 ? Int64(0) : mark === nothing ? nothing : unrealized_pnl(p, mark)
    equity =
        unrealized === nothing ? nothing :
        _checked_i64(Int128(p.cash_subunits) + p.total_cost_subunits + unrealized, "equity")
    equity !== nothing && (state.peak = max(state.peak, equity))
    drawdown = equity === nothing ? Int64(0) : state.peak - equity
    state.step += 1
    push!(
        state.ledger,
        LedgerPoint(
            state.step,
            ts,
            kind,
            ref,
            p.cash_subunits,
            p.inventory_lots,
            p.total_cost_subunits,
            p.realized_pnl_subunits,
            p.fees_subunits,
            mark,
            status,
            unrealized,
            equity,
            state.peak,
            drawdown,
            length(state.fills),
        ),
    )
    return nothing
end

# ── Driver ───────────────────────────────────────────────────────────────────

function _assumptions(config::SimulationConfig, portfolio::PaperPortfolio, session::ReplaySession)
    model = config.fill_model
    return Dict{String, Any}(
        "schema" => "dendritetrader.paper_sim.v1",
        "paper_only" => true,
        "venue" => String(session.venue),
        "instrument" => session.instrument,
        "allow_sequence_gaps" => session.policy.allow_sequence_gaps,
        "latency" => Dict{String, Any}(
            "model" => "fixed",
            "order_entry_ns" => config.latency.order_entry_ns,
            "cancel_ns" => config.latency.cancel_ns,
            "report_ns" => config.latency.report_ns,
        ),
        "feed" => _feed_record(config.feed),
        "fill_model" => Dict{String, Any}(
            "model" => "l2_queue_approximation",
            "approximation" => true,
            "queue_position" => String(model.queue_position),
            "taker_slippage_ticks" => model.taker_slippage_ticks,
            "seed" => Int64(model.seed % UInt64(typemax(Int64))),
            "rules" => collect(_FILL_ASSUMPTIONS),
        ),
        "fees" => Dict{String, Any}(
            "maker_fee_ppm" => config.fees.maker_fee_ppm,
            "taker_fee_ppm" => config.fees.taker_fee_ppm,
            "per_fill_fee_subunits" => config.fees.per_fill_fee_subunits,
            "sign" => "positive=commission, negative=rebate; applied once per fill",
            "rounding" => "commission ceil, rebate toward zero",
        ),
        "limits" => Dict{String, Any}(
            "max_order_lots" => config.limits.max_order_lots,
            "max_inventory_lots" => config.limits.max_inventory_lots,
            "max_initial_cash_subunits" => config.limits.max_initial_cash_subunits,
            "max_open_orders" => config.limits.max_open_orders,
            "scope" => "single instrument; not a multi-asset portfolio cap",
        ),
        "accounting" => Dict{String, Any}(
            "mode" => String(portfolio.mode),
            "parity_subset" => portfolio.mode == :long_flat,
            "price_scale" => portfolio.scale.price_scale,
            "qty_scale" => portfolio.scale.qty_scale,
            "mark" => "venue mid at or before the step, half tick rounded against the position",
        ),
        "event_order" => "time, then venue < activation < cancel < observation, then id",
    )
end

"""
    simulate!(portfolio, session, decide; config) -> SimulationResult

Run a deterministic paper simulation of `session` and commit the resulting state to
`portfolio` only if the whole run succeeds (any error leaves `portfolio` unchanged).

`decide(ctx::DecisionContext)` is called once per observed market event, at the time
the event becomes observable under `config.feed`, and returns `nothing`, an
`OrderIntent`/`CancelIntent`, or a vector of them. Orders are risk-checked at the
decision time (a rejection mutates nothing but the audit trail) and activated at
`decision_ts + config.latency.order_entry_ns`.

Requires `portfolio.instrument == session.instrument`, `qty_scale == 1` (so every
partial fill notional is exact) and initial cash within `config.limits`.
"""
function simulate!(
    portfolio::PaperPortfolio,
    session::ReplaySession,
    decide;
    config::SimulationConfig,
)
    _check_instrument(portfolio, session.instrument)
    portfolio.scale.qty_scale == 1 ||
        _fail(:unsupported_subset, "the simulator requires qty_scale == 1")
    0 <= portfolio.cash_subunits <= config.limits.max_initial_cash_subunits ||
        throw(ArgumentError("initial cash must be within [0, max_initial_cash_subunits]"))
    abs(portfolio.inventory_lots) <= config.limits.max_inventory_lots ||
        throw(ArgumentError("initial inventory exceeds max_inventory_lots"))
    _assert_label_free(decide)

    events = session.events
    available = [available_time(config.feed, event) for event in events]
    for index in eachindex(events)
        available[index] >= events[index].exchange_ts_ns ||
            throw(ArgumentError("an event cannot be observable before it occurs"))
        index == 1 ||
            available[index] >= available[index - 1] ||
            throw(ArgumentError("feed availability times must not decrease"))
    end

    working = copy(portfolio)
    start_equity = _checked_i64(
        Int128(working.cash_subunits) + working.total_cost_subunits,
        "starting book value",
    )
    state = _SimState(
        config,
        working,
        working.cash_subunits,
        working.inventory_lots,
        OrderBookState(session.venue, session.instrument),
        OrderBookState(session.venue, session.instrument),
        Dict{Tuple{BookSide, Int64}, Int64}(),
        PaperOrder[],
        Fill[],
        Int64[],
        Dict{Int64, Vector{Int}}(),
        _Action[],
        0,
        DecisionRecord[],
        LedgerPoint[],
        nothing,
        start_equity,
        0,
        0,
        nothing,
    )

    venue_index = 1
    observe_index = 1
    n = length(events)
    while true
        candidates = Tuple{Tuple{Int64, Int, Int64}, Symbol}[]
        venue_index <= n && push!(
            candidates,
            (
                (events[venue_index].exchange_ts_ns, _RANK_VENUE, events[venue_index].sequence),
                :venue,
            ),
        )
        isempty(state.actions) || push!(candidates, (_key(first(state.actions)), :action))
        observe_index <= n && push!(
            candidates,
            (
                (available[observe_index], _RANK_OBSERVATION, events[observe_index].sequence),
                :observe,
            ),
        )
        isempty(candidates) && break
        (ts, _, ref), kind = minimum(candidates)
        if kind == :venue
            event = events[venue_index]
            venue_index += 1
            apply!(state.venue, event; policy = session.policy)
            event isa BookDelta ? _on_book_delta!(state, event) : _on_trade_print!(state, event)
            _mark_point!(state, ts, :venue_event, event.sequence)
        elseif kind == :action
            action = popfirst!(state.actions)
            order = state.orders[action.order_id]
            if action.rank == _RANK_ACTIVATION
                _activate!(state, order, ts)
                _mark_point!(state, ts, :activation, order.order_id)
            else
                _is_live(order) && _terminate!(state, order, :cancelled, :cancel, ts, "user_cancel")
                _mark_point!(state, ts, :cancel, order.order_id)
            end
        else
            event = events[observe_index]
            observe_index += 1
            _observe!(state, session, event, ts, decide)
            _mark_point!(state, ts, :decision, state.decision_counter)
        end
    end

    end_ts = isempty(state.ledger) ? Int64(0) : last(state.ledger).ts_ns
    for order in state.orders
        _is_live(order) || continue
        _terminate!(state, order, :expired, :non_fill, end_ts, "session_end_unfilled")
    end
    _mark_point!(state, end_ts, :session_end, 0)

    assumptions = _assumptions(config, working, session)
    metrics = _metrics(state)
    payload = _canonical_payload(
        assumptions,
        state.decisions,
        state.orders,
        state.fills,
        state.ledger,
        working.audit,
        metrics,
    )
    digest = bytes2hex(sha256(payload))
    result = SimulationResult(
        assumptions,
        state.decisions,
        [copy(order) for order in state.orders],
        state.fills,
        state.ledger,
        copy(working.audit),
        metrics,
        in_parity_subset(working),
        portfolio.cash_subunits,
        portfolio.inventory_lots,
        portfolio.total_cost_subunits,
        working.instrument,
        working.scale,
        digest,
    )
    _commit_portfolio!(portfolio, working)
    return result
end

# ── Forecast-driven example policy (consumes Task 4 forecasts) ───────────────

"""
    ForecastPolicy(model, horizon; threshold, order_lots, max_inventory_lots)

Minimal long/flat example policy built on a fitted Task 4 `AbstractForecastModel`. At
each observed complete book it predicts one `Forecast` from the causal feature row
only. If `P(Up) >= threshold` and the reported inventory has room it sends an IOC buy
limited at the observed best ask; if `P(Down) >= threshold` and inventory is long it
sends an IOC sell of the reported inventory limited at the observed best bid. It does
nothing while it has an unreported working order. It is a wiring example, not a
strategy claim.
"""
struct ForecastPolicy{M <: AbstractForecastModel}
    model::M
    horizon::ForecastHorizon
    threshold::Float64
    order_lots::Int64
    max_inventory_lots::Int64
    function ForecastPolicy(
        model::M,
        horizon::ForecastHorizon;
        threshold::Real = 0.5,
        order_lots::Integer = 1,
        max_inventory_lots::Integer = 1,
    ) where {M <: AbstractForecastModel}
        0 < threshold <= 1 || throw(ArgumentError("threshold must be in (0, 1]"))
        order_lots > 0 || throw(ArgumentError("order_lots must be positive"))
        max_inventory_lots > 0 || throw(ArgumentError("max_inventory_lots must be positive"))
        return new{M}(model, horizon, Float64(threshold), order_lots, max_inventory_lots)
    end
end

function (policy::ForecastPolicy)(ctx::DecisionContext)
    ctx.features === nothing && return nothing
    isempty(ctx.open_orders) || return nothing
    forecast = only(predict!(policy.model, FeatureFrame([ctx.features]), policy.horizon))
    p_down, _, p_up = forecast.probabilities
    inventory = ctx.reported_inventory_lots
    if p_up >= policy.threshold && inventory + policy.order_lots <= policy.max_inventory_lots
        return OrderIntent(
            Buy,
            policy.order_lots,
            first(first(ctx.snapshot.asks));
            time_in_force = :ioc,
            tag = "forecast_up",
        )
    elseif p_down >= policy.threshold && inventory > 0
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

end # module
