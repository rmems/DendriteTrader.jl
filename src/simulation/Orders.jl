# SPDX-License-Identifier: MIT OR Apache-2.0

const _TIME_IN_FORCE = (:gtc, :ioc)

"""
    OrderIntent(side, qty_lots, limit_price_ticks; time_in_force=:gtc, post_only=false, tag="")

A hypothetical limit order requested by a decision function. Every simulated order is
a limit order: a "market" order must state an explicit protection limit, so worst-case
cash use is bounded before the order is accepted. `side` is `Buy` or `Sell`; quantities
are integer lots and prices are integer ticks. `time_in_force` is `:gtc` (rest until
filled or cancelled) or `:ioc` (cancel any remainder after the activation-time match).
`post_only` orders are rejected, not repriced, if they would take liquidity.

An intent is never sent anywhere. It only enters the paper simulator.
"""
struct OrderIntent
    side::TradeSide
    qty_lots::Int64
    limit_price_ticks::Int64
    time_in_force::Symbol
    post_only::Bool
    tag::String

    function OrderIntent(
        side::TradeSide,
        qty_lots::Integer,
        limit_price_ticks::Integer;
        time_in_force::Symbol = :gtc,
        post_only::Bool = false,
        tag::AbstractString = "",
    )
        side in (Buy, Sell) || throw(ArgumentError("order side must be Buy or Sell"))
        0 < qty_lots <= typemax(Int64) || throw(ArgumentError("qty_lots must be positive"))
        0 < limit_price_ticks <= typemax(Int64) ||
            throw(ArgumentError("limit_price_ticks must be positive"))
        time_in_force in _TIME_IN_FORCE ||
            throw(ArgumentError("time_in_force must be :gtc or :ioc"))
        post_only &&
            time_in_force == :ioc &&
            throw(ArgumentError("a post-only IOC order can never execute"))
        return new(
            side,
            Int64(qty_lots),
            Int64(limit_price_ticks),
            time_in_force,
            post_only,
            String(tag),
        )
    end
end

"""
    CancelIntent(order_id)

Request cancellation of a previously accepted paper order. The cancel takes effect
only after `FixedLatency.cancel_ns`; the order can still fill in the meantime.
"""
struct CancelIntent
    order_id::Int64
    function CancelIntent(order_id::Integer)
        order_id > 0 || throw(ArgumentError("order_id must be positive"))
        return new(Int64(order_id))
    end
end

"""
    RiskLimits(; max_order_lots, max_inventory_lots, max_initial_cash_subunits, max_open_orders)

Bounds enforced before any order is accepted. Inventory is checked against the worst
case in which every live order fills. These limits apply to the single simulated
instrument only; they are not a multi-asset portfolio cap.
"""
struct RiskLimits
    max_order_lots::Int64
    max_inventory_lots::Int64
    max_initial_cash_subunits::Int64
    max_open_orders::Int

    function RiskLimits(;
        max_order_lots::Integer = 100,
        max_inventory_lots::Integer = 100,
        max_initial_cash_subunits::Integer = 1_000_000_000,
        max_open_orders::Integer = 16,
    )
        max_order_lots > 0 || throw(ArgumentError("max_order_lots must be positive"))
        max_inventory_lots > 0 || throw(ArgumentError("max_inventory_lots must be positive"))
        max_initial_cash_subunits > 0 ||
            throw(ArgumentError("max_initial_cash_subunits must be positive"))
        max_open_orders > 0 || throw(ArgumentError("max_open_orders must be positive"))
        return new(
            Int64(max_order_lots),
            Int64(max_inventory_lots),
            Int64(max_initial_cash_subunits),
            Int(max_open_orders),
        )
    end
end

"""
    PaperOrder

Simulator-owned lifecycle of one accepted (or rejected) `OrderIntent`.

Four clocks are kept separate:

- `info_available_ts_ns`: when the newest event the decision used became observable
  under the configured feed-delay model (`info_sequence` identifies that event);
- `decision_ts_ns`: when the decision function ran (equal to the information time;
  compute latency is part of `FixedLatency.order_entry_ns`);
- `activation_ts_ns`: `decision_ts_ns + order_entry_ns`, the earliest venue time at
  which the order can interact with the book;
- each `Fill.exchange_ts_ns`, which is never earlier than `activation_ts_ns`.

`status` is one of `:pending`, `:active`, `:filled`, `:cancelled`, `:rejected`, `:expired`
(still working when the session ended).
`queue_ahead_lots` (current) and `initial_queue_ahead_lots` (when it started resting) are
L2 approximations, not observed queue positions.
"""
mutable struct PaperOrder
    order_id::Int64
    decision_id::Int64
    intent::OrderIntent
    info_sequence::Int64
    info_available_ts_ns::Int64
    decision_ts_ns::Int64
    activation_ts_ns::Int64
    status::Symbol
    remaining_lots::Int64
    filled_lots::Int64
    queue_ahead_lots::Int64
    initial_queue_ahead_lots::Int64
    resting::Bool
    terminal_ts_ns::Union{Nothing, Int64}
    terminal_reason::Union{Nothing, String}
end

Base.copy(order::PaperOrder) = PaperOrder(
    order.order_id,
    order.decision_id,
    order.intent,
    order.info_sequence,
    order.info_available_ts_ns,
    order.decision_ts_ns,
    order.activation_ts_ns,
    order.status,
    order.remaining_lots,
    order.filled_lots,
    order.queue_ahead_lots,
    order.initial_queue_ahead_lots,
    order.resting,
    order.terminal_ts_ns,
    order.terminal_reason,
)

_is_live(order::PaperOrder) = order.status == :pending || order.status == :active
_order_ref(run::Integer, order_id::Integer) = "r$(run)-o$(order_id)"

"""
Strategy-visible view of a working order. Fill progress is the *reported* progress:
only fills whose report time (`exchange_ts_ns + FixedLatency.report_ns`) is at or
before the decision time are included, so a fill cannot reveal venue state before
the market-data feed does. An order (including a rejected one) leaves the view only
once `terminal_ts_ns + report_ns` is at or before the decision time.
"""
struct OpenOrderView
    order_id::Int64
    side::TradeSide
    limit_price_ticks::Int64
    qty_lots::Int64
    reported_filled_lots::Int64
end
