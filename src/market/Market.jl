# SPDX-License-Identifier: MIT OR Apache-2.0

module Market

using JSON
using ..DendriteTrader: TradeSide, Buy, Sell, Neutral

export BookSide, Bid, Ask
export MarketEvent, BookDelta, TradePrint, BookSnapshot
export OrderBookState, ReplayPolicy, ReplaySession
export apply!, snapshot, best_bid, best_ask, spread_ticks, replay!, load_session_jsonl

include("Events.jl")
include("OrderBook.jl")
include("Replay.jl")

end # module
