# SPDX-License-Identifier: MIT OR Apache-2.0

# Run-local state only, never shared across tasks. Linked queues let terminal orders
# leave in O(1), without retaining tombstones on the event/decision traversal paths.
mutable struct _OrderLinks
    head::Int64
    tail::Int64
    previous::Vector{Int64}
    next::Vector{Int64}
end
_OrderLinks() = _OrderLinks(0, 0, Int64[], Int64[])

function _link!(links::_OrderLinks, id::Int64)
    links.tail == 0 ? (links.head = id) : (links.next[links.tail] = id)
    links.previous[id] = links.tail
    links.next[id] = 0
    links.tail = id
    return nothing
end

function _unlink!(links::_OrderLinks, id::Int64)
    before, after = links.previous[id], links.next[id]
    before == 0 ? (links.head = after) : (links.next[before] = after)
    after == 0 ? (links.tail = before) : (links.previous[after] = before)
    return nothing
end

mutable struct _RestingLevel
    head::Int64
    tail::Int64
    # Fenwick sums of remaining lots by level-local join slot. Deleting or filling
    # an earlier order updates every later order's own-ahead total in O(log joins).
    lots::Vector{Int64}
end
_RestingLevel() = _RestingLevel(0, 0, Int64[])

function _prefix(level::_RestingLevel, slot::Int)
    total = Int64(0)
    while slot > 0
        total += level.lots[slot]
        slot -= slot & -slot
    end
    return total
end

function _append_lots!(level::_RestingLevel, qty::Int64)
    slot = length(level.lots) + 1
    low = slot - (slot & -slot)
    push!(level.lots, qty + _prefix(level, slot - 1) - _prefix(level, low))
    return slot
end

function _change_lots!(level::_RestingLevel, slot::Int, delta::Int64)
    while slot <= length(level.lots)
        level.lots[slot] += delta
        slot += slot & -slot
    end
    return nothing
end

struct _Report
    ts::Int64
    fill_index::Int
    terminal_id::Int64
end

mutable struct _SimIndexes
    levels::Dict{Tuple{TradeSide, Int64}, _RestingLevel}
    prices::Dict{TradeSide, Vector{Int64}}
    resting::_OrderLinks
    slots::Vector{Int}
    visible::_OrderLinks
    reported_lots::Vector{Int64}
    reports::Vector{_Report}
    report_cursor::Int
    inventory::Int64
    cash::Int64
    live_count::Int
    live_buy::Int64
    live_sell::Int64
    reserved_cash::Int128
end

_SimIndexes(cash, inventory) = _SimIndexes(
    Dict{Tuple{TradeSide, Int64}, _RestingLevel}(),
    Dict(Buy => Int64[], Sell => Int64[]),
    _OrderLinks(),
    Int[],
    _OrderLinks(),
    Int64[],
    _Report[],
    1,
    inventory,
    cash,
    0,
    0,
    0,
    0,
)

function _index_order!(indexes::_SimIndexes, id::Int64)
    for links in (indexes.visible, indexes.resting)
        push!(links.previous, 0)
        push!(links.next, 0)
    end
    push!(indexes.slots, 0)
    push!(indexes.reported_lots, 0)
    _link!(indexes.visible, id)
    return nothing
end

function _join_level!(indexes::_SimIndexes, order::PaperOrder)
    side, price = order.intent.side, order.intent.limit_price_ticks
    key = (side, price)
    level = get(indexes.levels, key, nothing)
    if level === nothing
        level = _RestingLevel()
        indexes.levels[key] = level
        prices = indexes.prices[side]
        insert!(prices, searchsortedfirst(prices, price), price)
    end
    id = order.order_id
    links = indexes.resting
    links.previous[id] = level.tail
    level.tail == 0 ? (level.head = id) : (links.next[level.tail] = id)
    level.tail = id
    indexes.slots[id] = _append_lots!(level, order.remaining_lots)
    return nothing
end

function _leave_level!(indexes::_SimIndexes, order::PaperOrder)
    key = (order.intent.side, order.intent.limit_price_ticks)
    level = indexes.levels[key]
    id = order.order_id
    links = indexes.resting
    before, after = links.previous[id], links.next[id]
    before == 0 ? (level.head = after) : (links.next[before] = after)
    after == 0 ? (level.tail = before) : (links.previous[after] = before)
    if level.head == 0
        delete!(indexes.levels, key)
        prices = indexes.prices[order.intent.side]
        deleteat!(prices, searchsortedfirst(prices, order.intent.limit_price_ticks))
    end
    return nothing
end
