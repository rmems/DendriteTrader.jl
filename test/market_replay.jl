# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using DendriteTrader

@testset "Market replay API" begin
    required = (
        :BookSide,
        :Bid,
        :Ask,
        :MarketEvent,
        :BookDelta,
        :TradePrint,
        :BookSnapshot,
        :OrderBookState,
        :ReplayPolicy,
        :ReplaySession,
        :apply!,
        :snapshot,
        :best_bid,
        :best_ask,
        :spread_ticks,
        :replay!,
        :load_session_jsonl,
    )
    for name in required
        @test isdefined(DendriteTrader, name)
    end
end

const FIXTURE = joinpath(@__DIR__, "fixtures", "book_session.jsonl")

@testset "Market event validation" begin
    @test_throws ArgumentError OrderBookState(Symbol(""), "XYZ")
    @test_throws ArgumentError OrderBookState(:SIM, "")

    valid_delta = BookDelta(:SIM, "XYZ", 1, 2, 1, Bid, 100, 10)
    @test valid_delta isa MarketEvent
    @test_throws ArgumentError BookDelta(Symbol(""), "XYZ", 1, 2, 1, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "", 1, 2, 1, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 0, 2, 1, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, 0, 1, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 2, 1, 1, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, 2, 0, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, 2, 1, Bid, 0, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, 2, 1, Bid, 100, 0)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", true, 2, 1, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, true, 1, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, 2, true, Bid, 100, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, 2, 1, Bid, true, 10)
    @test_throws ArgumentError BookDelta(:SIM, "XYZ", 1, 2, 1, Bid, 100, true)

    valid_trade = TradePrint(:SIM, "XYZ", 1, 2, 1, Buy, 100, 3)
    @test valid_trade isa MarketEvent
    @test_throws ArgumentError TradePrint(:SIM, "XYZ", 1, 2, 1, Neutral, 100, 3)
    @test_throws ArgumentError TradePrint(:SIM, "XYZ", 1, 2, 1, Buy, 100, 0)
    @test_throws ArgumentError TradePrint(:SIM, "XYZ", true, 2, 1, Buy, 100, 3)
    @test_throws ArgumentError TradePrint(:SIM, "XYZ", 1, true, 1, Buy, 100, 3)
    @test_throws ArgumentError TradePrint(:SIM, "XYZ", 1, 2, true, Buy, 100, 3)
    @test_throws ArgumentError TradePrint(:SIM, "XYZ", 1, 2, 1, Buy, true, 3)
    @test_throws ArgumentError TradePrint(:SIM, "XYZ", 1, 2, 1, Buy, 100, true)
end

@testset "JSONL fixture loading" begin
    events = load_session_jsonl(FIXTURE)
    @test events isa ReplaySession
    @test length(events) == 6
    @test events[1] == BookDelta(:SIM, "XYZ", 1_000, 1_010, 1, Bid, 100, 10)
    @test events[2] == BookDelta(:SIM, "XYZ", 1_000, 1_011, 2, Ask, 102, 12)
    @test events[4] == TradePrint(:SIM, "XYZ", 1_002, 1_013, 4, Buy, 102, 3)

    mktemp() do path, io
        write(io, "{\"type\":\"mystery\"}\n")
        close(io)
        @test_throws ArgumentError load_session_jsonl(path)
    end

    mktemp() do path, io
        write(
            io,
            "{\"type\":\"book_delta\",\"venue\":\"SIM\",\"instrument\":\"XYZ\",\"exchange_ts_ns\":100,\"receive_ts_ns\":110,\"sequence\":1,\"side\":\"BID\",\"price_ticks\":100,\"size_delta\":10}\n",
        )
        write(
            io,
            "{\"type\":\"book_delta\",\"venue\":\"UNTRUSTED-VENUE\",\"instrument\":\"XYZ\",\"exchange_ts_ns\":200,\"receive_ts_ns\":210,\"sequence\":2,\"side\":\"ASK\",\"price_ticks\":102,\"size_delta\":10}\n",
        )
        close(io)

        error = try
            load_session_jsonl(path)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("invalid event at line 2", sprint(showerror, error))
    end
end

@testset "Deterministic L2 replay" begin
    events = load_session_jsonl(FIXTURE)
    book = OrderBookState(:SIM, "XYZ")
    snapshots = replay!(book, events)

    @test length(snapshots) == length(events)
    @test snapshots[2].bids == [100 => 10]
    @test snapshots[2].asks == [102 => 12]
    @test snapshots[3].bids == [101 => 7, 100 => 10]
    @test snapshots[4].bids == snapshots[3].bids
    @test snapshots[4].sequence == 4
    @test snapshots[5].bids == [101 => 5, 100 => 10]
    @test snapshots[6].bids == [100 => 10]
    @test best_bid(book) == (100 => 10)
    @test best_ask(book) == (102 => 12)
    @test spread_ticks(book) == 2
    @test snapshot(book; depth = 0).bids == Pair{Int64, Int64}[]
end

@testset "Replay ordering and state integrity" begin
    first = BookDelta(:SIM, "XYZ", 100, 110, 1, Bid, 100, 10)

    @testset "sequence and time" begin
        book = OrderBookState(:SIM, "XYZ")
        apply!(book, first)
        @test_throws ArgumentError apply!(book, BookDelta(:SIM, "XYZ", 101, 111, 1, Ask, 102, 10))
        @test_throws ArgumentError apply!(book, BookDelta(:SIM, "XYZ", 99, 112, 2, Ask, 102, 10))
        @test_throws ArgumentError apply!(book, BookDelta(:SIM, "XYZ", 102, 109, 2, Ask, 102, 10))
        @test_throws ArgumentError apply!(book, BookDelta(:SIM, "XYZ", 102, 113, 3, Ask, 102, 10))
        @test book.sequence == 1

        apply!(
            book,
            BookDelta(:SIM, "XYZ", 102, 113, 3, Ask, 102, 10);
            policy = ReplayPolicy(allow_sequence_gaps = true),
        )
        @test book.sequence == 3
    end

    @testset "same timestamp is causal when sequence increases" begin
        book = OrderBookState(:SIM, "XYZ")
        apply!(book, first)
        apply!(book, BookDelta(:SIM, "XYZ", 100, 111, 2, Ask, 102, 10))
        @test book.sequence == 2
    end

    @testset "invalid events do not partially mutate" begin
        book = OrderBookState(:SIM, "XYZ")
        apply!(book, first)
        before = snapshot(book)

        @test_throws ArgumentError apply!(book, BookDelta(:OTHER, "XYZ", 101, 111, 2, Ask, 102, 10))
        @test snapshot(book) == before
        @test_throws ArgumentError apply!(book, BookDelta(:SIM, "OTHER", 101, 111, 2, Ask, 102, 10))
        @test snapshot(book) == before
        @test_throws ArgumentError apply!(book, BookDelta(:SIM, "XYZ", 101, 111, 2, Bid, 100, -11))
        @test snapshot(book) == before
        @test_throws ArgumentError apply!(book, BookDelta(:SIM, "XYZ", 101, 111, 2, Ask, 99, 10))
        @test snapshot(book) == before
    end

    @testset "trade prints advance causality without mutating depth" begin
        book = OrderBookState(:SIM, "XYZ")
        apply!(book, first)
        depth_before = (copy(book.bids), copy(book.asks))
        apply!(book, TradePrint(:SIM, "XYZ", 101, 111, 2, Sell, 100, 2))
        @test (book.bids, book.asks) == depth_before
        @test book.sequence == 2
    end

    @testset "batch replay is transactional" begin
        book = OrderBookState(:SIM, "XYZ")
        before = snapshot(book)
        valid = BookDelta(:SIM, "XYZ", 100, 110, 1, Bid, 100, 10)
        gap = BookDelta(:SIM, "XYZ", 101, 111, 3, Ask, 102, 10)

        @test_throws ArgumentError replay!(book, MarketEvent[valid]; depth = -1)
        @test snapshot(book) == before
        @test_throws ArgumentError replay!(book, ReplaySession(MarketEvent[valid, gap]))
        @test snapshot(book) == before

        snapshots = replay!(book, MarketEvent[valid])
        @test book.sequence == 1
        @test snapshots[end] == snapshot(book)
    end
end
