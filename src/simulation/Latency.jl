# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    FixedLatency(order_entry_ns; cancel_ns=order_entry_ns, report_ns=order_entry_ns)

Deterministic decision-to-venue latency. An order decided at `t` becomes active at
`t + order_entry_ns` and cannot interact with any venue book state before then
(venue events stamped exactly at the activation time are applied first). A cancel
decided at `t` takes effect at `t + cancel_ns`; the order may fill in between. A fill
or terminal order state at venue time `u` becomes visible to the decision function at
`u + report_ns`. Decision compute time is folded into `order_entry_ns`. Zero latency is
allowed as an explicit diagnostic, not a realistic claim.
"""
struct FixedLatency
    order_entry_ns::Int64
    cancel_ns::Int64
    report_ns::Int64
    function FixedLatency(
        order_entry_ns::Integer;
        cancel_ns::Integer = order_entry_ns,
        report_ns::Integer = order_entry_ns,
    )
        0 <= order_entry_ns <= typemax(Int64) ||
            throw(ArgumentError("order_entry_ns must be non-negative"))
        0 <= cancel_ns <= typemax(Int64) || throw(ArgumentError("cancel_ns must be non-negative"))
        0 <= report_ns <= typemax(Int64) || throw(ArgumentError("report_ns must be non-negative"))
        return new(Int64(order_entry_ns), Int64(cancel_ns), Int64(report_ns))
    end
end

_checked_time(base::Int64, delta::Int64) = Base.Checked.checked_add(base, delta)

"""Venue time at which an order decided at `decision_ts_ns` becomes active."""
activation_time(latency::FixedLatency, decision_ts_ns::Integer) =
    _checked_time(Int64(decision_ts_ns), latency.order_entry_ns)

"""Venue time at which a cancel decided at `decision_ts_ns` takes effect."""
cancel_effective_time(latency::FixedLatency, decision_ts_ns::Integer) =
    _checked_time(Int64(decision_ts_ns), latency.cancel_ns)

"""Feed-delay model deciding when a market event becomes observable to a decision."""
abstract type AbstractFeedDelay end

"""Events become observable at their recorded `receive_ts_ns` (the default)."""
struct ReceiveTimestampFeed <: AbstractFeedDelay end

"""Events become observable at `exchange_ts_ns + delay_ns`."""
struct FixedFeedDelay <: AbstractFeedDelay
    delay_ns::Int64
    function FixedFeedDelay(delay_ns::Integer)
        0 <= delay_ns <= typemax(Int64) || throw(ArgumentError("delay_ns must be non-negative"))
        return new(Int64(delay_ns))
    end
end

"""Time at which `event` becomes observable under `feed`."""
available_time(::ReceiveTimestampFeed, event::MarketEvent) = event.receive_ts_ns
available_time(feed::FixedFeedDelay, event::MarketEvent) =
    _checked_time(event.exchange_ts_ns, feed.delay_ns)

_feed_record(::ReceiveTimestampFeed) =
    Dict{String, Any}("model" => "receive_timestamp", "delay_ns" => nothing)
_feed_record(feed::FixedFeedDelay) =
    Dict{String, Any}("model" => "fixed_exchange_delay", "delay_ns" => feed.delay_ns)
