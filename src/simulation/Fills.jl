# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    FeeSchedule(; maker_fee_ppm=0, taker_fee_ppm=0, per_fill_fee_subunits=0)

Integer fee convention, applied exactly once per `Fill` when the fill is created and
stored in `Fill.fee_subunits` (quote subunits). Rates are parts per million of the
fill notional. A positive fee is a commission; a negative rate is a rebate.

Rounding is adverse to the simulated trader: commissions round up (`cld`), rebates
round toward zero. `per_fill_fee_subunits` is a non-negative flat charge per fill.
"""
struct FeeSchedule
    maker_fee_ppm::Int64
    taker_fee_ppm::Int64
    per_fill_fee_subunits::Int64
    function FeeSchedule(;
        maker_fee_ppm::Integer = 0,
        taker_fee_ppm::Integer = 0,
        per_fill_fee_subunits::Integer = 0,
    )
        # Explicit bounds on the original integer: `abs(typemin(Int64))` overflows.
        -1_000_000 <= maker_fee_ppm <= 1_000_000 ||
            throw(ArgumentError("maker_fee_ppm must be within ±1_000_000"))
        -1_000_000 <= taker_fee_ppm <= 1_000_000 ||
            throw(ArgumentError("taker_fee_ppm must be within ±1_000_000"))
        0 <= per_fill_fee_subunits <= typemax(Int32) ||
            throw(ArgumentError("per_fill_fee_subunits must be non-negative"))
        return new(Int64(maker_fee_ppm), Int64(taker_fee_ppm), Int64(per_fill_fee_subunits))
    end
end

function _rate_fee(notional::Int64, ppm::Int64)
    scaled = Int128(notional) * abs(ppm)
    return ppm >= 0 ? cld(scaled, 1_000_000) : -fld(scaled, 1_000_000)
end

"""Fee in quote subunits for one fill of `notional` with `liquidity` `:maker` or `:taker`."""
function fill_fee(fees::FeeSchedule, notional::Integer, liquidity::Symbol)
    liquidity in (:maker, :taker) || throw(ArgumentError("liquidity must be :maker or :taker"))
    ppm = liquidity == :maker ? fees.maker_fee_ppm : fees.taker_fee_ppm
    return _checked_i64(_rate_fee(Int64(notional), ppm) + fees.per_fill_fee_subunits, "fee")
end

# Worst-case (largest) positive fee, used only to reserve cash before acceptance.
_max_fee(fees::FeeSchedule, notional::Int64) =
    max(Int64(0), fill_fee(fees, notional, :maker), fill_fee(fees, notional, :taker))

"""
    L2QueueApproximation(; queue_position=:back, taker_slippage_ticks=0, seed=0)

Declared, **approximate** fill model for L2 (price-level aggregated) replay data.
Price-level data does not reveal queue position, so every passive fill below is an
inference, not an observed fill (compare the hftbacktest order-fill documentation).

Assumptions, all recorded in `SimulationResult.assumptions`:

1. Activation: an order can interact only with venue state at or after
   `decision_ts + order_entry_ns`; venue events stamped at the activation instant are
   applied first. It never trades against a print that occurred before activation.
2. Taking: a marketable order sweeps *displayed* opposite levels inside its limit at
   each level's price (plus `taker_slippage_ticks` adverse ticks), as taker.
3. Depletion: displayed liquidity the simulator consumed stays unavailable at that
   price until the replayed level is fully removed; new displayed size adds back.
4. Joining: a resting order joins the **back** of its price level
   (`queue_position = :back`), or a seeded uniform position in `[0, displayed]` when
   `queue_position = :random` (less conservative, labelled as such).
5. Queue advance (risk-averse): only trade prints at the order's price advance the
   queue; displayed-size decreases advance it only when the level becomes smaller
   than the remaining venue queue ahead. Simulated orders at one level form a FIFO
   queue in join order: an order's queue ahead is its venue-ahead estimate plus the
   remaining size of simulated orders that joined earlier, and venue cancels never
   reduce the simulated part. Print volume that filled simulated orders at their price
   is treated as venue size that would still be resting (no market impact), both when
   later orders join and when capping on displayed-size decreases. A print fills only
   the size left after the queue.
6. Price priority: a print walks the hit side from the best price down to the print
   price. At each level it consumes that level's whole estimated queue (venue lots and
   simulated orders in FIFO order, including venue lots our orders absorbed earlier)
   before reaching worse levels. A print strictly through a resting limit can therefore
   fill it only after better-priced and earlier-queued size is used up.
7. Quote-through: opposite displayed liquidity appearing strictly through a resting
   limit fills it as maker at the limit price, up to that displayed size, better-priced
   resting orders first, then join order. A locked
   quote (equal price) does not fill.
8. No market impact: hypothetical fills do not alter the replayed book beyond the
   depletion bookkeeping above. No matching-engine guarantee is implied.
"""
struct L2QueueApproximation
    queue_position::Symbol
    taker_slippage_ticks::Int64
    seed::UInt64
    function L2QueueApproximation(;
        queue_position::Symbol = :back,
        taker_slippage_ticks::Integer = 0,
        seed::Integer = 0,
    )
        queue_position in (:back, :random) ||
            throw(ArgumentError("queue_position must be :back or :random"))
        0 <= taker_slippage_ticks <= typemax(Int32) ||
            throw(ArgumentError("taker_slippage_ticks must be non-negative"))
        seed >= 0 || throw(ArgumentError("seed must be non-negative"))
        return new(queue_position, Int64(taker_slippage_ticks), UInt64(seed))
    end
end

const _FILL_ASSUMPTIONS = (
    "L2 queue position is inferred, not observed; passive fills are approximations",
    "orders interact only with venue state at or after decision_ts + order_entry_ns",
    "venue events stamped at an activation instant are applied before the activation",
    "taker orders sweep displayed levels within the limit at level price plus slippage",
    "simulator-consumed displayed liquidity is unavailable until the level is removed",
    "queue advances only on prints at the order price, or when the level shrinks below it",
    "simulated orders at one level are FIFO by join order; venue cancels never skip them",
    "print volume absorbed by simulated orders stays as resting venue size at that level",
    "prints walk the hit side best price first; each level's whole FIFO queue is consumed first",
    "displayed liquidity strictly through a resting limit fills it as maker at the limit, best price first",
    "a locked quote at the resting limit does not fill",
    "no market impact; no venue matching-engine guarantee",
)

"""
    Fill

One simulated (hypothetical) execution. `exchange_ts_ns` is the venue time of the fill
and is never earlier than `activation_ts_ns = decision_ts_ns + order_entry_ns`.
`price_ticks` already includes any declared slippage (`slippage_subunits`, positive is
adverse) and `fee_subunits` is the single fee charge for this fill (positive commission,
negative rebate). The portfolio applies `fee_subunits` once and adds nothing else.
`trigger` is `:activation_sweep`, `:trade_print` or `:quote_through`.
"""
struct Fill
    fill_id::String
    order_id::Int64
    decision_id::Int64
    side::TradeSide
    price_ticks::Int64
    qty_lots::Int64
    liquidity::Symbol
    fee_subunits::Int64
    slippage_subunits::Int64
    exchange_ts_ns::Int64
    report_ts_ns::Int64
    activation_ts_ns::Int64
    decision_ts_ns::Int64
    info_available_ts_ns::Int64
    info_sequence::Int64
    trigger::Symbol
    trigger_sequence::Int64
end

# ── Venue-liquidity helpers (operate on the simulator's venue book) ──────────

_levels(book::OrderBookState, side::BookSide) = side == Bid ? book.bids : book.asks

function _effective_size(book::OrderBookState, depletion, side::BookSide, price::Int64)
    displayed = get(_levels(book, side), price, Int64(0))
    return max(Int64(0), displayed - get(depletion, (side, price), Int64(0)))
end

function _deplete!(depletion, side::BookSide, price::Int64, qty::Int64)
    depletion[(side, price)] = get(depletion, (side, price), Int64(0)) + qty
    return depletion
end

# Opposite-side levels a `side` order could take, best first.
function _opposite_levels(book::OrderBookState, side::TradeSide)
    if side == Buy
        return sort!(collect(keys(book.asks)))
    else
        return sort!(collect(keys(book.bids)); rev = true)
    end
end

_opposite_book_side(side::TradeSide) = side == Buy ? Ask : Bid
_own_book_side(side::TradeSide) = side == Buy ? Bid : Ask

_within_limit(side::TradeSide, price::Int64, limit::Int64) =
    side == Buy ? price <= limit : price >= limit
_strictly_through(side::TradeSide, price::Int64, limit::Int64) =
    side == Buy ? price < limit : price > limit
_slipped(side::TradeSide, price::Int64, slip::Int64) = side == Buy ? price + slip : price - slip

"""True if `order` would take liquidity against displayed opposite size right now."""
function _is_marketable(book, depletion, model::L2QueueApproximation, intent::OrderIntent)
    opposite = _opposite_book_side(intent.side)
    for price in _opposite_levels(book, intent.side)
        fill_price = _slipped(intent.side, price, model.taker_slippage_ticks)
        _within_limit(intent.side, fill_price, intent.limit_price_ticks) || return false
        _effective_size(book, depletion, opposite, price) > 0 && return true
    end
    return false
end
