# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    PortfolioError(kind, message)

Typed paper-accounting failure. `kind` uses the stable machine names of the
`metabolic-ledger.parity.v1` contract: `:unsupported_short`, `:instrument_mismatch`,
`:malformed_input`, `:out_of_range`, `:duplicate_fill_conflict`, `:unsupported_subset`.
A call that throws leaves the portfolio exactly as it was.
"""
struct PortfolioError <: Exception
    kind::Symbol
    message::String
end

Base.showerror(io::IO, error::PortfolioError) =
    print(io, "PortfolioError(", error.kind, "): ", error.message)

_fail(kind::Symbol, message::AbstractString) = throw(PortfolioError(kind, String(message)))

function _checked_i64(value::Integer, what::AbstractString)
    typemin(Int64) <= value <= typemax(Int64) || _fail(:out_of_range, "$what overflowed Int64")
    return Int64(value)
end

_add(a::Int64, b::Int64, what) = _checked_i64(Int128(a) + Int128(b), what)
_sub(a::Int64, b::Int64, what) = _checked_i64(Int128(a) - Int128(b), what)

"""
    ParityScale(price_scale, qty_scale)

Integer unit convention shared with `metabolic-ledger`: `price_scale` ticks are one
quote unit and `qty_scale` lots are one asset unit. Cash, fees, cost and PnL are quote
subunits, and `notional = price_ticks * qty_lots / qty_scale` must be exact.
"""
struct ParityScale
    price_scale::Int64
    qty_scale::Int64
    function ParityScale(price_scale::Integer, qty_scale::Integer)
        price_scale > 0 || _fail(:malformed_input, "price_scale must be positive")
        qty_scale > 0 || _fail(:malformed_input, "qty_scale must be positive")
        return new(Int64(price_scale), Int64(qty_scale))
    end
end

ParityScale(; price_scale::Integer = 100, qty_scale::Integer = 1) =
    ParityScale(price_scale, qty_scale)

"""Exact notional in quote subunits for a positive price and quantity."""
function notional_subunits(scale::ParityScale, price_ticks::Integer, qty_lots::Integer)
    price_ticks > 0 || _fail(:malformed_input, "price_ticks must be positive")
    qty_lots > 0 || _fail(:malformed_input, "qty_lots must be positive")
    product = Int128(price_ticks) * Int128(qty_lots)
    product % scale.qty_scale == 0 || _fail(
        :malformed_input,
        "qty_scale $(scale.qty_scale) does not divide price_ticks * qty_lots $product",
    )
    return _checked_i64(div(product, scale.qty_scale), "notional")
end

# Signed mark value of a (possibly short) inventory; zero when flat.
function _signed_mark_value(scale::ParityScale, mark_price_ticks::Integer, inventory::Int64)
    inventory == 0 && return Int64(0)
    value = notional_subunits(scale, mark_price_ticks, abs(inventory))
    return inventory > 0 ? value : -value
end

"""Payload identity of an applied fill: time, instrument, side, price, qty, fee."""
const _FillFingerprint = Tuple{Int64, String, TradeSide, Int64, Int64, Int64}

"""One audit row, field-for-field the `metabolic-ledger` `AuditRecord`."""
struct AuditRow
    event_time_ns::Int64
    kind::Symbol
    fill_id::Union{Nothing, String}
    order_id::Union{Nothing, String}
    instrument::String
    side::Union{Nothing, TradeSide}
    price_ticks::Union{Nothing, Int64}
    qty_lots::Union{Nothing, Int64}
    fee_subunits::Union{Nothing, Int64}
    realized_pnl_delta::Int64
    cash_after::Int64
    inventory_after::Int64
    total_cost_after::Int64
    reason::Union{Nothing, String}
    idempotent_replay::Bool
end

_side_string(side::TradeSide) = side == Buy ? "buy" : side == Sell ? "sell" : "neutral"
_side_string(::Nothing) = nothing

"""Contract-shaped dictionary of an audit row (used for parity comparison and export)."""
audit_dict(row::AuditRow) = Dict{String, Any}(
    "event_time_ns" => row.event_time_ns,
    "kind" => String(row.kind),
    "fill_id" => row.fill_id,
    "order_id" => row.order_id,
    "instrument" => row.instrument,
    "side" => _side_string(row.side),
    "price_ticks" => row.price_ticks,
    "qty_lots" => row.qty_lots,
    "fee_subunits" => row.fee_subunits,
    "realized_pnl_delta" => row.realized_pnl_delta,
    "cash_after" => row.cash_after,
    "inventory_after" => row.inventory_after,
    "total_cost_after" => row.total_cost_after,
    "reason" => row.reason,
    "idempotent_replay" => row.idempotent_replay,
)

"""
    PaperPortfolio(instrument, initial_cash_subunits; scale=ParityScale(), mode=:long_flat)

Single-instrument integer evaluation state for the paper simulator. It is a
**reference/evaluation** state for experiments, not an authoritative or persistent
ledger; persistent accounting belongs to `metabolic-ledger`.

`mode = :long_flat` is the declared cross-language parity subset (inventory never
negative; a sell beyond inventory is `:unsupported_short`, never clamped).
`mode = :signed` is a Julia-only research extension that allows short inventory and
position flips; results in that mode are outside the parity subset and are labelled so.

Accounting (identical to the `metabolic-ledger` accounting contract in the long/flat
subset):

- Buy cash change `-(notional + fee)`; sell cash change `+(notional - fee)`.
- `fee > 0` is a commission, `fee < 0` a rebate; each fill's fee is applied once.
- `total_cost_subunits` holds the signed position cost; buy fees enter the basis.
- A reduction allocates `trunc(total_cost * qty / |inventory|)` toward zero, and a full
  close allocates the remaining cost. Realized PnL is `cash_flow - allocated`.
- Invariant: `cash + total_cost == initial_cash + realized_pnl`.

`simulation_runs` counts `simulate!` runs committed to this portfolio. `simulate!`
only accepts a fresh portfolio (zero runs, empty audit); the `r<run>-` prefix on order
and fill ids is kept as defence in depth against id collisions.
"""
mutable struct PaperPortfolio
    instrument::String
    scale::ParityScale
    mode::Symbol
    initial_cash_subunits::Int64
    cash_subunits::Int64
    inventory_lots::Int64
    total_cost_subunits::Int64
    realized_pnl_subunits::Int64
    fees_subunits::Int64
    applied::Dict{String, _FillFingerprint}
    audit::Vector{AuditRow}
    simulation_runs::Int
end

function PaperPortfolio(
    instrument::AbstractString,
    initial_cash_subunits::Integer;
    scale::ParityScale = ParityScale(),
    mode::Symbol = :long_flat,
)
    isempty(instrument) && _fail(:malformed_input, "instrument must be non-empty")
    mode in (:long_flat, :signed) ||
        _fail(:unsupported_subset, "mode must be :long_flat or :signed")
    cash = _checked_i64(initial_cash_subunits, "initial cash")
    return PaperPortfolio(
        String(instrument),
        scale,
        mode,
        cash,
        cash,
        0,
        0,
        0,
        0,
        Dict{String, _FillFingerprint}(),
        AuditRow[],
        0,
    )
end

Base.copy(p::PaperPortfolio) = PaperPortfolio(
    p.instrument,
    p.scale,
    p.mode,
    p.initial_cash_subunits,
    p.cash_subunits,
    p.inventory_lots,
    p.total_cost_subunits,
    p.realized_pnl_subunits,
    p.fees_subunits,
    copy(p.applied),
    copy(p.audit),
    p.simulation_runs,
)

# `source` is a private working copy that is discarded after the commit, so sharing its
# `applied`/`audit` containers with `target` is safe (nothing else holds them).
function _commit_portfolio!(target::PaperPortfolio, source::PaperPortfolio)
    for name in fieldnames(PaperPortfolio)
        setfield!(target, name, getfield(source, name))
    end
    return target
end

"""True when the portfolio is in the declared long/flat parity subset."""
in_parity_subset(p::PaperPortfolio) = p.mode == :long_flat

"""Floor average entry in ticks per asset unit (`0` when flat); Euclidean floor."""
function weighted_cost_basis_ticks(p::PaperPortfolio)
    p.inventory_lots == 0 && return Int64(0)
    return _checked_i64(
        fld(Int128(p.total_cost_subunits) * p.scale.qty_scale, Int128(p.inventory_lots)),
        "weighted cost basis",
    )
end

"""Unrealized PnL against an external integer mark; `0` when flat. Never stored."""
function unrealized_pnl(p::PaperPortfolio, mark_price_ticks::Integer)
    p.inventory_lots == 0 && return Int64(0)
    mark_price_ticks > 0 || _fail(:malformed_input, "mark_price_ticks must be positive")
    return _sub(
        _signed_mark_value(p.scale, mark_price_ticks, p.inventory_lots),
        p.total_cost_subunits,
        "unrealized_pnl",
    )
end

function _check_instrument(p::PaperPortfolio, instrument::AbstractString)
    instrument == p.instrument || _fail(
        :instrument_mismatch,
        "portfolio holds $(p.instrument), event requested $(instrument)",
    )
    return nothing
end

"""
    apply_fill!(portfolio, fill_id, event_time_ns, instrument, side, price_ticks, qty_lots, fee_subunits)

Apply one already-decided fill exactly once. The same `fill_id` with an identical
payload is an idempotent replay (audited, no second debit); a different payload is
`:duplicate_fill_conflict`. Returns `:applied` or `:idempotent_replay`. Validation and
the full post-state are computed before any field changes.
"""
function apply_fill!(
    p::PaperPortfolio,
    fill_id::AbstractString,
    event_time_ns::Integer,
    instrument::AbstractString,
    side::TradeSide,
    price_ticks::Integer,
    qty_lots::Integer,
    fee_subunits::Integer,
)
    isempty(fill_id) && _fail(:malformed_input, "fill_id must be non-empty")
    event_time_ns >= 0 || _fail(:malformed_input, "event_time_ns must be non-negative")
    _checked_i64(event_time_ns, "event_time_ns")
    _check_instrument(p, instrument)
    side in (Buy, Sell) || _fail(:malformed_input, "fill side must be Buy or Sell")
    notional = notional_subunits(p.scale, price_ticks, qty_lots)
    fee = _checked_i64(fee_subunits, "fee")
    fingerprint =
        (Int64(event_time_ns), String(instrument), side, Int64(price_ticks), Int64(qty_lots), fee)

    previous = get(p.applied, fill_id, nothing)
    if previous !== nothing
        previous == fingerprint ||
            _fail(:duplicate_fill_conflict, "fill_id $(fill_id) reused with a different payload")
        push!(
            p.audit,
            AuditRow(
                Int64(event_time_ns),
                :fill,
                String(fill_id),
                nothing,
                p.instrument,
                side,
                Int64(price_ticks),
                Int64(qty_lots),
                fee,
                0,
                p.cash_subunits,
                p.inventory_lots,
                p.total_cost_subunits,
                "idempotent_replay",
                true,
            ),
        )
        return :idempotent_replay
    end

    qty = Int64(qty_lots)
    signed_qty = side == Buy ? qty : -qty
    # Positive cash flow is an inflow. Fee sign: commission reduces cash, rebate adds.
    cash_flow = side == Buy ? -_add(notional, fee, "buy debit") : _sub(notional, fee, "sell credit")

    inventory = p.inventory_lots
    new_inventory = _add(inventory, signed_qty, "inventory")
    if p.mode == :long_flat && new_inventory < 0
        _fail(
            :unsupported_short,
            "sell $(qty) lots of $(p.instrument) with inventory $(inventory) (long/flat subset)",
        )
    end

    reducing = inventory != 0 && sign(signed_qty) != sign(inventory)
    close_qty = reducing ? min(qty, abs(inventory)) : Int64(0)
    open_qty = qty - close_qty

    # Split the single fee between the closing and opening legs of a flip (signed mode
    # only); the two legs always sum to the original fee, so it is still applied once.
    close_flow = cash_flow
    open_flow = Int64(0)
    if close_qty > 0 && open_qty > 0
        close_notional = notional_subunits(p.scale, price_ticks, close_qty)
        close_fee = _checked_i64(div(Int128(fee) * close_qty, qty), "close fee")
        close_flow =
            side == Buy ? -_add(close_notional, close_fee, "close debit") :
            _sub(close_notional, close_fee, "close credit")
        open_flow = _sub(cash_flow, close_flow, "open flow")
    elseif close_qty == 0
        close_flow = Int64(0)
        open_flow = cash_flow
    end

    total_cost = p.total_cost_subunits
    realized_delta = Int64(0)
    if close_qty > 0
        allocated = if close_qty == abs(inventory)
            total_cost
        else
            _checked_i64(div(Int128(total_cost) * close_qty, abs(inventory)), "allocated cost")
        end
        realized_delta = _sub(close_flow, allocated, "realized delta")
        total_cost = _sub(total_cost, allocated, "total_cost")
    end
    if open_qty > 0
        total_cost = _sub(total_cost, open_flow, "total_cost")
    end
    new_inventory == 0 && total_cost != 0 && _fail(:out_of_range, "flat book retained cost")

    new_cash = _add(p.cash_subunits, cash_flow, "cash")
    new_realized = _add(p.realized_pnl_subunits, realized_delta, "realized_pnl")
    new_fees = _add(p.fees_subunits, fee, "fees")

    p.cash_subunits = new_cash
    p.inventory_lots = new_inventory
    p.total_cost_subunits = total_cost
    p.realized_pnl_subunits = new_realized
    p.fees_subunits = new_fees
    p.applied[String(fill_id)] = fingerprint
    push!(
        p.audit,
        AuditRow(
            Int64(event_time_ns),
            :fill,
            String(fill_id),
            nothing,
            p.instrument,
            side,
            Int64(price_ticks),
            qty,
            fee,
            realized_delta,
            new_cash,
            new_inventory,
            total_cost,
            nothing,
            false,
        ),
    )
    return :applied
end

"""
    record_non_fill!(portfolio, kind, order_id, event_time_ns, instrument, reason)

Audit a `:reject`, `:cancel` or `:non_fill`. Cash, inventory, cost and PnL never change.
"""
function record_non_fill!(
    p::PaperPortfolio,
    kind::Symbol,
    order_id::AbstractString,
    event_time_ns::Integer,
    instrument::AbstractString,
    reason::AbstractString,
)
    kind in (:reject, :cancel, :non_fill) ||
        _fail(:malformed_input, "non-fill kind must be :reject, :cancel or :non_fill")
    event_time_ns >= 0 || _fail(:malformed_input, "event_time_ns must be non-negative")
    time = _checked_i64(event_time_ns, "event_time_ns")
    _check_instrument(p, instrument)
    isempty(order_id) && _fail(:malformed_input, "order_id must be non-empty")
    push!(
        p.audit,
        AuditRow(
            time,
            kind,
            nothing,
            String(order_id),
            p.instrument,
            nothing,
            nothing,
            nothing,
            nothing,
            0,
            p.cash_subunits,
            p.inventory_lots,
            p.total_cost_subunits,
            String(reason),
            false,
        ),
    )
    return p
end

"""`cash + total_cost == initial_cash + realized` — the per-event accounting identity."""
reconciles(p::PaperPortfolio) =
    Int128(p.cash_subunits) + p.total_cost_subunits ==
    Int128(p.initial_cash_subunits) + p.realized_pnl_subunits
