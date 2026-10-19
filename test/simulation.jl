# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using JSON
using SHA
using DendriteTrader

const _SIM_VENUE = :SIM
const _SIM_INSTRUMENT = "ASSET_A"
const _PARITY_DIR = joinpath(@__DIR__, "fixtures", "metabolic_ledger_parity_v1")

# Event builders. `recv` defaults to the exchange time (zero feed delay).
_bd(seq, ts, side, price, size; recv = ts) =
    BookDelta(_SIM_VENUE, _SIM_INSTRUMENT, ts, recv, seq, side, price, size)
_tp(seq, ts, aggressor, price, size; recv = ts) =
    TradePrint(_SIM_VENUE, _SIM_INSTRUMENT, ts, recv, seq, aggressor, price, size)

_session(events) = ReplaySession(events)
_portfolio(; cash = 100_000, mode = :long_flat) =
    PaperPortfolio(_SIM_INSTRUMENT, cash; scale = ParityScale(100, 1), mode = mode)
_config(latency; kwargs...) = SimulationConfig(; latency = FixedLatency(latency), kwargs...)

# Decide exactly once, on the observation of `sequence`; otherwise do nothing.
function _at_sequence(actions::Dict{Int, Any})
    return ctx -> get(actions, Int(ctx.info_sequence), nothing)
end
_at_sequence(sequence::Integer, output) = _at_sequence(Dict{Int, Any}(Int(sequence) => output))

_fill_ts_ok(result, latency) =
    all(f -> f.exchange_ts_ns >= f.decision_ts_ns + latency, result.fills) &&
    all(f -> f.exchange_ts_ns >= f.activation_ts_ns, result.fills)

# Two-sided base book: bid 100 x 10, ask 102 x 10 at t = 100.
_base_events() = MarketEvent[_bd(1, 100, Bid, 100, 10), _bd(2, 100, Ask, 102, 10)]

@testset "Paper simulation (RM-1864)" begin
    @testset "Contracts validate inputs" begin
        @test_throws ArgumentError OrderIntent(Neutral, 1, 100)
        @test_throws ArgumentError OrderIntent(Buy, 0, 100)
        @test_throws ArgumentError OrderIntent(Buy, 1, 0)
        @test_throws ArgumentError OrderIntent(Buy, 1, 100; time_in_force = :fok)
        @test_throws ArgumentError OrderIntent(Buy, 1, 100; time_in_force = :ioc, post_only = true)
        @test_throws ArgumentError FixedLatency(-1)
        @test_throws ArgumentError FixedFeedDelay(-1)
        @test_throws ArgumentError L2QueueApproximation(queue_position = :front)
        @test_throws ArgumentError FeeSchedule(taker_fee_ppm = 2_000_000)
        @test_throws ArgumentError RiskLimits(max_order_lots = 0)
        @test_throws PortfolioError ParityScale(0, 1)
        @test_throws PortfolioError PaperPortfolio("", 0)
        @test_throws PortfolioError PaperPortfolio(_SIM_INSTRUMENT, 0; mode = :margin)
        latency = FixedLatency(25; cancel_ns = 5, report_ns = 7)
        @test activation_time(latency, 100) == 125
        @test cancel_effective_time(latency, 100) == 105
        @test_throws OverflowError activation_time(latency, typemax(Int64))
        @test available_time(FixedFeedDelay(9), _bd(1, 100, Bid, 100, 1; recv = 105)) == 109
        @test available_time(ReceiveTimestampFeed(), _bd(1, 100, Bid, 100, 1; recv = 105)) == 105
    end

    @testset "Separate information, decision, activation and fill clocks" begin
        events = MarketEvent[
            _bd(1, 100, Bid, 100, 10; recv = 104),
            _bd(2, 100, Ask, 102, 10; recv = 106),
            _bd(3, 140, Bid, 100, 1; recv = 141),
        ]
        portfolio = _portfolio()
        result = simulate!(
            portfolio,
            _session(events),
            _at_sequence(2, OrderIntent(Buy, 3, 102; time_in_force = :ioc));
            config = _config(20),
        )
        decision = only(result.decisions)
        order = only(result.orders)
        fill = only(result.fills)
        # Information about sequence 2 is available at its receive time (106), the
        # decision happens then, the order activates 20 ns later, and fills no earlier.
        @test decision.info_sequence == 2
        @test decision.info_available_ts_ns == 106
        @test decision.decision_ts_ns == 106
        @test decision.effective_ts_ns == 126
        @test order.activation_ts_ns == 126
        @test fill.exchange_ts_ns == 126
        @test fill.info_available_ts_ns == 106
        @test fill.decision_ts_ns == 106
        @test fill.activation_ts_ns == 126
        @test _fill_ts_ok(result, 20)
        @test result.assumptions["latency"]["order_entry_ns"] == 20
        @test result.assumptions["feed"]["model"] == "receive_timestamp"
        @test isempty(reconciliation_errors(result))
    end

    @testset "Decision contexts only contain events observable under the feed model" begin
        events = MarketEvent[
            _bd(1, 100, Bid, 100, 10),
            _bd(2, 101, Ask, 102, 10),
            _bd(3, 105, Ask, 102, -10),
            _bd(4, 106, Ask, 103, 10),
            _bd(5, 200, Bid, 100, 1),
        ]
        for feed in (ReceiveTimestampFeed(), FixedFeedDelay(0), FixedFeedDelay(7))
            seen = Tuple{Int64, Int64, Int64, Vector{Pair{Int64, Int64}}}[]
            decide =
                ctx -> begin
                    push!(
                        seen,
                        (
                            ctx.decision_ts_ns,
                            ctx.info_sequence,
                            ctx.snapshot.sequence,
                            copy(ctx.snapshot.asks),
                        ),
                    )
                    nothing
                end
            simulate!(_portfolio(), _session(events), decide; config = _config(0; feed = feed))
            @test length(seen) == length(events)
            for (decision_ts, info_sequence, snapshot_sequence, _) in seen
                # The snapshot reflects exactly the newest observable event; no event whose
                # availability time is after the decision time is included.
                @test snapshot_sequence == info_sequence
                for event in events
                    if event.sequence > info_sequence
                        @test available_time(feed, event) >= decision_ts
                    else
                        @test available_time(feed, event) <= decision_ts
                    end
                end
            end
            # With a 7 ns feed delay the decision for sequence 2 happens at 108, after the
            # venue already removed the ask at 105, yet the context still shows the ask.
            if feed isa FixedFeedDelay && feed.delay_ns == 7
                decision_ts, _, _, asks = seen[2]
                @test decision_ts == 108
                @test asks == [102 => 10]
            end
        end
        decreasing = MarketEvent[
            _bd(1, 100, Bid, 100, 10; recv = 150),
            _bd(2, 110, Ask, 102, 10; recv = 150),
        ]
        @test simulate!(_portfolio(), _session(decreasing), _ -> nothing; config = _config(0)) isa
              SimulationResult
    end

    @testset "Spread crossing fills as taker at displayed prices" begin
        events =
            vcat(_base_events(), MarketEvent[_bd(3, 100, Ask, 103, 10), _bd(4, 200, Bid, 100, 1)])
        portfolio = _portfolio()
        result = simulate!(
            portfolio,
            _session(events),
            _at_sequence(3, OrderIntent(Buy, 12, 103; time_in_force = :ioc));
            config = _config(10),
        )
        @test [(f.price_ticks, f.qty_lots, f.liquidity) for f in result.fills] == [(102, 10, :taker), (103, 2, :taker)]
        @test only(result.orders).status == :filled
        @test portfolio.inventory_lots == 12
        @test portfolio.cash_subunits == 100_000 - (102 * 10 + 103 * 2)
        @test isempty(reconciliation_errors(result))
        # A limit below the ask does not cross.
        limited = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(3, OrderIntent(Buy, 12, 101; time_in_force = :ioc));
            config = _config(10),
        )
        @test isempty(limited.fills)
        @test only(limited.orders).status == :cancelled
    end

    @testset "Passive resting order without trades does not fill" begin
        events =
            vcat(_base_events(), MarketEvent[_bd(3, 150, Bid, 100, 5), _bd(4, 200, Bid, 100, -5)])
        portfolio = _portfolio()
        result = simulate!(
            portfolio,
            _session(events),
            _at_sequence(2, OrderIntent(Buy, 4, 100));
            config = _config(10),
        )
        order = only(result.orders)
        @test isempty(result.fills)
        @test order.status == :expired
        @test order.queue_ahead_lots == 10
        @test portfolio.cash_subunits == 100_000
        @test last(result.audit).kind == :non_fill
        @test last(result.audit).reason == "session_end_unfilled"
    end

    @testset "Partial fill leaves an IOC remainder cancelled and a GTC remainder resting" begin
        events = vcat(_base_events(), MarketEvent[_bd(3, 200, Bid, 100, 1)])
        ioc = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(2, OrderIntent(Buy, 15, 102; time_in_force = :ioc));
            config = _config(10),
        )
        @test only(ioc.fills).qty_lots == 10
        @test only(ioc.orders).status == :cancelled
        @test only(ioc.orders).terminal_reason == "ioc_remainder"
        @test only(ioc.orders).filled_lots == 10
        gtc = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(2, OrderIntent(Buy, 15, 102));
            config = _config(10),
        )
        @test only(gtc.fills).qty_lots == 10
        @test only(gtc.orders).remaining_lots == 5
        @test only(gtc.orders).status == :expired
        @test only(gtc.orders).queue_ahead_lots == 0
        @test gtc.metrics.fill_rate == 10 / 15
    end

    @testset "Depleted displayed liquidity is not reused" begin
        events = vcat(
            _base_events(),
            MarketEvent[
                _bd(3, 150, Bid, 100, 1),
                _bd(4, 300, Ask, 102, 3),
                _bd(5, 400, Bid, 100, 1),
            ],
        )
        two_orders = Dict{Int, Any}(
            2 => [
                OrderIntent(Buy, 6, 102; time_in_force = :ioc),
                OrderIntent(Buy, 6, 102; time_in_force = :ioc),
            ],
            3 => OrderIntent(Buy, 6, 102; time_in_force = :ioc),
            4 => OrderIntent(Buy, 6, 102; time_in_force = :ioc),
        )
        result =
            simulate!(_portfolio(), _session(events), _at_sequence(two_orders); config = _config(5))
        # 10 displayed: first order takes 6, second takes the remaining 4. A later order
        # sees zero (consumed) liquidity until the replay adds 3 new lots.
        @test [(f.order_id, f.qty_lots) for f in result.fills] == [(1, 6), (2, 4), (4, 3)]
        @test result.orders[3].status == :cancelled
        @test isempty(reconciliation_errors(result))
    end

    @testset "Cancellation takes effect only after cancel latency" begin
        events = vcat(
            _base_events(),
            MarketEvent[
                _bd(3, 120, Bid, 101, 0 + 1),
                _bd(4, 130, Bid, 101, -1),
                _tp(5, 132, Sell, 101, 1),
                _tp(6, 150, Sell, 101, 5),
            ],
        )
        # Rest a bid at 101 (improving the spread, queue 0 at activation 110). Cancel at
        # 130 with 10 ns cancel latency: the print at 132 still fills; the one at 150 not.
        actions = Dict{Int, Any}(2 => OrderIntent(Buy, 3, 101), 4 => CancelIntent(1))
        latency = FixedLatency(10; cancel_ns = 10)
        result = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(actions);
            config = SimulationConfig(; latency = latency),
        )
        @test [(f.exchange_ts_ns, f.qty_lots) for f in result.fills] == [(132, 1)]
        order = only(result.orders)
        @test order.status == :cancelled
        @test order.terminal_ts_ns == 140
        cancel = result.decisions[2]
        @test cancel.action == :cancel && cancel.effective_ts_ns == 140
        @test any(row -> row.kind == :cancel && row.reason == "user_cancel", result.audit)
        # Cancelling an unknown order is a rejected decision with no state change.
        unknown = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(2, CancelIntent(99));
            config = _config(10),
        )
        @test !only(unknown.decisions).accepted
        @test only(unknown.audit).kind == :reject
    end

    @testset "Deterministic event ordering and replay hashes" begin
        events = vcat(
            _base_events(),
            MarketEvent[
                _tp(3, 150, Sell, 100, 12),
                _tp(4, 160, Sell, 100, 30),
                _bd(5, 200, Bid, 100, 1),
            ],
        )
        actions = Dict{Int, Any}(2 => [OrderIntent(Buy, 4, 100), OrderIntent(Buy, 4, 100)])
        runs = [
            simulate!(_portfolio(), _session(events), _at_sequence(actions); config = _config(10)) for _ in 1:3
        ]
        @test length(unique(run.result_hash for run in runs)) == 1
        @test length(runs[1].result_hash) == 64
        # Same activation instant: order 1 queues ahead of order 2 (id order).
        @test runs[1].orders[1].initial_queue_ahead_lots == 10
        @test runs[1].orders[2].initial_queue_ahead_lots == 14
        @test [(f.order_id, f.exchange_ts_ns, f.qty_lots) for f in runs[1].fills] == [(1, 150, 2), (1, 160, 2), (2, 160, 4)]
        # Seeded random queue placement is reproducible and recorded.
        seeded(seed) = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(actions);
            config = _config(
                10;
                fill_model = L2QueueApproximation(queue_position = :random, seed = seed),
            ),
        )
        @test seeded(7).result_hash == seeded(7).result_hash
        @test seeded(7).assumptions["fill_model"]["seed"] == "7"
        # The full 64-bit seed is recorded losslessly: seeds that differ only in the
        # high bit, and 0 vs typemax(Int64), give distinct assumptions and hashes.
        high = UInt64(1) << 63
        for (left, right) in ((5, high | 5), (0, typemax(Int64)), (0, typemax(UInt64)))
            @test seeded(left).assumptions["fill_model"]["seed"] !=
                  seeded(right).assumptions["fill_model"]["seed"]
            @test seeded(left).result_hash != seeded(right).result_hash
        end
        @test seeded(typemax(UInt64)).assumptions["fill_model"]["seed"] == "18446744073709551615"
        @test seeded(7).assumptions["fill_model"]["queue_position"] == "random"
        @test seeded(7).assumptions["fill_model"]["approximation"] == true
        @test seeded(7).result_hash != runs[1].result_hash
        # Changing a declared assumption changes the hash even if fills do not change.
        @test simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(actions);
            config = _config(11),
        ).result_hash != runs[1].result_hash
    end

    @testset "Leakage 1: queue position — no first-touch fill, no fill on the triggering print" begin
        events = MarketEvent[
            _bd(1, 100, Bid, 100, 10),
            _bd(2, 100, Ask, 102, 10),
            _tp(3, 200, Sell, 100, 3),   # the print whose information triggers the decision
            _bd(4, 201, Bid, 100, -3),   # displayed queue is now 7
            _tp(5, 300, Sell, 100, 7),   # first touch after activation: consumes the queue ahead
            _bd(6, 301, Bid, 100, -7),
            _tp(7, 400, Sell, 100, 2),
        ]
        for latency in (0, 1, 50)
            result = simulate!(
                _portfolio(),
                _session(events),
                _at_sequence(3, OrderIntent(Buy, 5, 100));
                config = _config(latency),
            )
            order = only(result.orders)
            @test order.decision_ts_ns == 200
            @test all(f -> f.trigger_sequence != 3, result.fills)
            @test all(f -> f.trigger_sequence != 5, result.fills)
            @test [(f.trigger_sequence, f.qty_lots) for f in result.fills] == [(7, 2)]
            @test _fill_ts_ok(result, latency)
        end
        # Improving the spread (empty queue at 101): the triggering print at 200 still
        # cannot fill; the first print after activation trades through 101 and does.
        improving = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(3, OrderIntent(Buy, 5, 101));
            config = _config(1),
        )
        @test only(improving.orders).initial_queue_ahead_lots == 0
        @test [(f.trigger_sequence, f.qty_lots) for f in improving.fills] == [(5, 5)]
    end

    @testset "Leakage 2: marks use only the venue mid at or before each event" begin
        prefix = MarketEvent[
            _bd(1, 100, Bid, 100, 10),
            _bd(2, 100, Ask, 102, 10),
            _bd(3, 150, Bid, 101, 4),
            _bd(4, 200, Ask, 103, 5),
        ]
        future_a = MarketEvent[_bd(5, 300, Ask, 102, -10), _bd(6, 310, Bid, 101, -4)]
        future_b = MarketEvent[_bd(5, 300, Ask, 102, -10), _bd(6, 310, Bid, 101, 6)]
        decide = _at_sequence(2, OrderIntent(Buy, 5, 102; time_in_force = :ioc))
        run(future) =
            simulate!(_portfolio(), _session(vcat(prefix, future)), decide; config = _config(10))
        a = run(future_a)
        b = run(future_b)
        cutoff = 300
        early_a = [p for p in a.ledger if p.ts_ns < cutoff]
        early_b = [p for p in b.ledger if p.ts_ns < cutoff]
        @test !isempty(early_a)
        @test early_a == early_b
        @test a.decisions == b.decisions
        @test a.fills == b.fills
        # The future perturbation does change later marks, so the test is sensitive.
        @test last(a.ledger).mark_price_ticks != last(b.ledger).mark_price_ticks
        @test last(a.ledger).equity_subunits != last(b.ledger).equity_subunits
        # Every fresh mark equals floor((best bid + best ask) / 2) of the venue book at
        # that step, rebuilt here independently from the events at or before the step.
        for point in a.ledger
            point.mark_status == :fresh || continue
            book = OrderBookState(_SIM_VENUE, _SIM_INSTRUMENT)
            for event in vcat(prefix, future_a)
                event.exchange_ts_ns <= point.ts_ns && apply!(book, event)
            end
            @test point.mark_price_ticks == fld(first(best_bid(book)) + first(best_ask(book)), 2)
        end
    end

    @testset "Leakage 3: no match against book states inside (t, t + L)" begin
        events = MarketEvent[
            _bd(1, 100, Bid, 100, 10),
            _bd(2, 100, Ask, 105, 10),
            _bd(3, 103, Ask, 101, 5),   # favourable ask appears inside (100, 110)
            _bd(4, 107, Ask, 101, -5),  # and is gone before t + L = 110
            _bd(5, 200, Bid, 100, 1),
        ]
        intent = OrderIntent(Buy, 5, 101; time_in_force = :ioc)
        slow =
            simulate!(_portfolio(), _session(events), _at_sequence(2, intent); config = _config(10))
        @test isempty(slow.fills)
        @test only(slow.orders).activation_ts_ns == 110
        @test only(slow.orders).status == :cancelled
        # Control: with L = 5 the order is active at 105 and does fill, proving the
        # scenario contains a fill that only latency removes.
        fast =
            simulate!(_portfolio(), _session(events), _at_sequence(2, intent); config = _config(5))
        @test [(f.exchange_ts_ns, f.price_ticks, f.qty_lots) for f in fast.fills] == [(105, 101, 5)]
        # Same for a resting order: a print inside (t, t+L) cannot fill it.
        prints = MarketEvent[
            _bd(1, 100, Bid, 100, 10),
            _bd(2, 100, Ask, 105, 10),
            _tp(3, 104, Sell, 101, 9),
            _tp(4, 115, Sell, 100, 1),
            _bd(5, 200, Bid, 100, 1),
        ]
        rest = simulate!(
            _portfolio(),
            _session(prints),
            _at_sequence(2, OrderIntent(Buy, 5, 101));
            config = _config(10),
        )
        # The 9-lot print inside (100, 110) fills nothing; only the post-activation
        # 1-lot trade-through at 115 does.
        @test [(f.trigger_sequence, f.qty_lots) for f in rest.fills] == [(4, 1)]
        # Fill reports are delayed too: reported inventory lags the venue fill.
        reported = Int64[]
        report_events =
            vcat(events[1:2], MarketEvent[_bd(3, 120, Bid, 100, 1), _bd(4, 190, Bid, 100, 1)])
        decide =
            ctx -> begin
                push!(reported, ctx.reported_inventory_lots)
                ctx.info_sequence == 2 ? OrderIntent(Buy, 2, 105; time_in_force = :ioc) : nothing
            end
        simulate!(
            _portfolio(),
            _session(report_events),
            decide;
            config = SimulationConfig(; latency = FixedLatency(10; report_ns = 50)),
        )
        @test reported == [0, 0, 0, 2]   # fill at 110 is reported at 160
    end

    @testset "Leakage 6: terminal order states are visible only after report latency" begin
        # Two sessions that differ only in venue state at/after the activation instant.
        # Every decision context observed before the outcome is reported must be identical.
        latency = FixedLatency(10; report_ns = 50)
        config = SimulationConfig(; latency = latency)
        function views(events, actions)
            seen = Tuple{Int64, Vector{OpenOrderView}, Int64}[]
            decide =
                ctx -> begin
                    push!(
                        seen,
                        (ctx.decision_ts_ns, copy(ctx.open_orders), ctx.reported_inventory_lots),
                    )
                    get(actions, Int(ctx.info_sequence), nothing)
                end
            result = simulate!(_portfolio(), _session(events), decide; config = config)
            return seen, result
        end
        quiet = MarketEvent[_bd(k, 100 + 5k, Bid, 90, 1) for k in 3:30]

        # (a) Post-only rejected at activation because the book crossed it in (t, t+L].
        crossed = vcat(_base_events(), MarketEvent[_bd(3, 105, Ask, 101, 5)], quiet[2:end])
        uncrossed = vcat(_base_events(), MarketEvent[_bd(3, 105, Ask, 103, 5)], quiet[2:end])
        actions = Dict{Int, Any}(2 => OrderIntent(Buy, 2, 101; post_only = true))
        seen_a, result_a = views(crossed, actions)
        seen_b, result_b = views(uncrossed, actions)
        @test only(result_a.orders).status == :rejected
        @test only(result_a.orders).terminal_reason == "post_only_would_take"
        @test only(result_a.orders).terminal_ts_ns == 110
        @test only(result_b.orders).status == :expired
        before = [v for v in seen_a if v[1] < 160]
        @test any(v -> 110 <= v[1] < 160, before)
        @test before == [v for v in seen_b if v[1] < 160]
        for (index, (ts, open_orders, _)) in enumerate(seen_a)
            index <= 2 && continue   # contexts at and before the submitting decision
            # Visible through [activation, activation + report_ns), gone afterwards.
            @test (ts < 160) == any(o -> o.order_id == 1, open_orders)
        end
        # Sensitivity: after the report the two runs are distinguishable.
        @test [v for v in seen_a if v[1] >= 160] != [v for v in seen_b if v[1] >= 160]

        # (b) A decision-time risk reject that depends on an unreported fill.
        limits = RiskLimits(max_inventory_lots = 6)
        liquid = vcat(_base_events(), MarketEvent[_bd(3, 105, Bid, 90, 1)], quiet[2:end])
        dry = vcat(
            MarketEvent[
                _bd(1, 100, Bid, 100, 10),
                _bd(2, 100, Ask, 102, 10),
                _bd(3, 105, Ask, 102, -10),
            ],
            quiet[2:end],
        )
        risk_actions = Dict{Int, Any}(
            2 => OrderIntent(Buy, 4, 102; time_in_force = :ioc),
            5 => OrderIntent(Buy, 4, 100),   # decided at 125, inside (110, 160)
        )
        function risk_views(events)
            seen = Tuple{Int64, Vector{OpenOrderView}, Int64}[]
            decide =
                ctx -> begin
                    push!(
                        seen,
                        (ctx.decision_ts_ns, copy(ctx.open_orders), ctx.reported_inventory_lots),
                    )
                    get(risk_actions, Int(ctx.info_sequence), nothing)
                end
            result = simulate!(
                _portfolio(),
                _session(events),
                decide;
                config = SimulationConfig(; latency = latency, limits = limits),
            )
            return seen, result
        end
        seen_l, result_l = risk_views(liquid)
        seen_d, result_d = risk_views(dry)
        @test result_l.decisions[2].reason == "inventory_limit"   # true inventory is 4
        @test result_d.decisions[2].accepted                      # IOC found no liquidity
        @test [v for v in seen_l if v[1] < 160] == [v for v in seen_d if v[1] < 160]
        @test [v for v in seen_l if v[1] >= 175] != [v for v in seen_d if v[1] >= 175]
    end

    @testset "Leakage 4: fees and slippage are applied exactly once" begin
        events =
            vcat(_base_events(), MarketEvent[_bd(3, 150, Bid, 100, 1), _bd(4, 300, Bid, 100, 1)])
        fees = FeeSchedule(taker_fee_ppm = 1_000, per_fill_fee_subunits = 1)
        actions = Dict{Int, Any}(
            2 => OrderIntent(Buy, 4, 103; time_in_force = :ioc),
            3 => OrderIntent(Sell, 4, 99; time_in_force = :ioc),
        )
        model = L2QueueApproximation(taker_slippage_ticks = 1)
        portfolio = _portfolio()
        result = simulate!(
            portfolio,
            _session(events),
            _at_sequence(actions);
            config = _config(10; fees = fees, fill_model = model),
        )
        buy, sell = result.fills
        # Analytic expectation: buy at ask 102 + 1 tick slippage, sell at bid 100 - 1.
        buy_notional = 103 * 4
        sell_notional = 99 * 4
        buy_fee = cld(buy_notional * 1_000, 1_000_000) + 1
        sell_fee = cld(sell_notional * 1_000, 1_000_000) + 1
        @test (buy.price_ticks, sell.price_ticks) == (103, 99)
        @test (buy.fee_subunits, sell.fee_subunits) == (buy_fee, sell_fee)
        @test buy.slippage_subunits == 4 && sell.slippage_subunits == 4
        analytic = (sell_notional - sell_fee) - (buy_notional + buy_fee)
        @test portfolio.realized_pnl_subunits == analytic
        @test portfolio.cash_subunits - 100_000 == analytic
        @test portfolio.fees_subunits == buy_fee + sell_fee
        @test result.metrics.slippage_subunits == 8
        # Double-counting (subtracting fees or slippage again) must not match.
        @test portfolio.realized_pnl_subunits != analytic - (buy_fee + sell_fee)
        @test portfolio.realized_pnl_subunits != analytic - result.metrics.slippage_subunits
        @test isempty(reconciliation_errors(result))
        # Tamper: a fill record whose fee differs from what the ledger charged is caught.
        tampered_fills = [
            Fill(
                (
                    f.fill_id,
                    f.order_id,
                    f.decision_id,
                    f.side,
                    f.price_ticks,
                    f.qty_lots,
                    f.liquidity,
                )...,
                2 * f.fee_subunits,
                (
                    f.slippage_subunits,
                    f.exchange_ts_ns,
                    f.report_ts_ns,
                    f.activation_ts_ns,
                    f.decision_ts_ns,
                    f.info_available_ts_ns,
                    f.info_sequence,
                    f.trigger,
                    f.trigger_sequence,
                )...,
            ) for f in result.fills
        ]
        tampered = SimulationResult(
            result.assumptions,
            result.decisions,
            result.orders,
            tampered_fills,
            result.ledger,
            result.audit,
            result.metrics,
            result.in_parity_subset,
            result.initial_cash_subunits,
            result.initial_inventory_lots,
            result.initial_total_cost_subunits,
            result.instrument,
            result.scale,
            result.result_hash,
        )
        @test !isempty(reconciliation_errors(tampered))
        # Rebate: a negative maker rate credits cash once.
        @test fill_fee(FeeSchedule(maker_fee_ppm = -100), 50_000, :maker) == -5
        @test fill_fee(FeeSchedule(maker_fee_ppm = -100), 49_999, :maker) == -4
        @test fill_fee(FeeSchedule(taker_fee_ppm = 100), 49_999, :taker) == 5
    end

    @testset "Leakage 5: decision functions cannot reach labels or outcomes" begin
        names = fieldnames(DecisionContext)
        @test !any(n -> occursin(r"label|target|outcome|future"i, String(n)), names)
        @test !any(T -> T <: MovementTarget, fieldtypes(DecisionContext))
        captured = Ref{Any}(nothing)
        events = vcat(_base_events(), MarketEvent[_bd(3, 200, Bid, 100, 1)])
        simulate!(
            _portfolio(),
            _session(events),
            ctx -> (captured[] = ctx; nothing);
            config = _config(0),
        )
        ctx = captured[]
        @test_throws ArgumentError ctx.label
        @test_throws ArgumentError ctx.target
        @test_throws ArgumentError ctx.outcome
        # Runtime guard: a closure that captures labels is refused before any mutation.
        labels = label_event_horizon(
            [
                BookSnapshot(:SIM, _SIM_INSTRUMENT, 100, 100, 1, [100 => 10], [102 => 10]),
                BookSnapshot(:SIM, _SIM_INSTRUMENT, 200, 200, 2, [101 => 10], [103 => 10]),
            ];
            horizon_events = 1,
        )
        portfolio = _portfolio()
        before = (portfolio.cash_subunits, length(portfolio.audit))
        peeking = ctx -> (first(labels).label == Up ? OrderIntent(Buy, 1, 102) : nothing)
        @test_throws ArgumentError simulate!(
            portfolio,
            _session(events),
            peeking;
            config = _config(0),
        )
        @test (portfolio.cash_subunits, length(portfolio.audit)) == before
        struct_peeker = (labels = Dict(1 => first(labels)),)
        @test_throws ArgumentError simulate!(
            portfolio,
            _session(events),
            ctx -> (struct_peeker.labels[1]; nothing);
            config = _config(0),
        )
    end

    @testset "Forecast policy consumes Task 4 forecasts from causal features only" begin
        events = MarketEvent[
            _bd(1, 100, Bid, 100, 30),
            _bd(2, 100, Ask, 102, 10),
            _bd(3, 200, Ask, 102, -5),
            _bd(4, 300, Bid, 100, -28),
            _bd(5, 400, Bid, 100, 1),
        ]
        model = ImbalanceRule(0.25)
        frame = FeatureFrame([
            FeatureRow(10, 1, 2, 101, 101, 0.5, 0),
            FeatureRow(20, 2, 2, 101, 101, -0.5, 0),
        ])
        fit!(model, frame, [MovementTarget(1, 2, 1, Down)])
        policy = ForecastPolicy(
            model,
            ForecastHorizon(1);
            threshold = 0.9,
            order_lots = 2,
            max_inventory_lots = 2,
        )
        portfolio = _portfolio()
        result = simulate!(portfolio, _session(events), policy; config = _config(10))
        @test [d.tag for d in result.decisions] == ["forecast_up", "forecast_down"]
        @test [f.side for f in result.fills] == [Buy, Sell]
        @test portfolio.inventory_lots == 0
        @test result.in_parity_subset
        @test isempty(reconciliation_errors(result))
    end

    @testset "Decision-path features equal offline causal features" begin
        session = load_session_jsonl(joinpath(@__DIR__, "fixtures", "book_session.jsonl"))
        offline = microstructure_features(
            replay!(OrderBookState(session.venue, session.instrument), session),
        )
        online = FeatureRow[]
        decide = ctx -> (ctx.features === nothing || push!(online, ctx.features); nothing)
        portfolio = PaperPortfolio(session.instrument, 1_000)
        simulate!(
            portfolio,
            session,
            decide;
            config = SimulationConfig(; latency = FixedLatency(0)),
        )
        @test online == collect(offline)
    end

    @testset "Existing backtester: same-tick execution at the triggering signal price" begin
        # Characterization of pre-existing code (not changed by RM-1864). run_backtest
        # executes every accepted signal at its own timestamp and price, with no latency,
        # queue or spread model. This is an optimistic zero-latency execution assumption
        # (the backtest consumes no market events, so no future book state is read), but
        # it is exactly what Leakage tests 1 and 3 forbid for the event-driven simulator.
        signals = [
            TradeSignal("MARKET-A", Buy, 100.0, 1.0, 0.95f0, 1_000),
            TradeSignal("MARKET-A", Sell, 101.0, 1.0, 0.95f0, 2_000),
        ]
        result =
            run_backtest(BacktestConfig(initial_balance = 10_000.0), signals; log_file = nothing)
        closed = only(filter(t -> t.is_closed, result.trade_log))
        @test closed.entry_time == signals[2].timestamp_ns
        @test closed.price == signals[2].price
        # A latency-aware backtester would report fills strictly after the decision time.
        @test_broken closed.entry_time > signals[2].timestamp_ns
    end

    @testset "Per-event reconciliation of cash, inventory, PnL and drawdown" begin
        events = vcat(
            _base_events(),
            MarketEvent[
                _bd(3, 150, Bid, 101, 5),
                _bd(4, 200, Bid, 101, -5),
                _bd(5, 210, Bid, 90, 10),
                _bd(6, 220, Bid, 100, -10),
                _tp(7, 300, Sell, 90, 20),
                _bd(8, 400, Ask, 95, 4),
                _bd(9, 500, Bid, 90, 1),
            ],
        )
        actions = Dict{Int, Any}(
            2 => OrderIntent(Buy, 6, 102; time_in_force = :ioc),
            3 => OrderIntent(Sell, 2, 101; time_in_force = :ioc),
            8 => OrderIntent(Buy, 1, 95; time_in_force = :ioc),
        )
        fees = FeeSchedule(maker_fee_ppm = -200, taker_fee_ppm = 500)
        result = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(actions);
            config = _config(5; fees = fees),
        )
        @test isempty(reconciliation_errors(result))
        @test length(result.fills) == 3
        @test all(
            p -> p.total_cost_subunits + p.cash_subunits == 100_000 + p.realized_pnl_subunits,
            result.ledger,
        )
        @test result.metrics.max_drawdown_subunits > 0
        @test result.metrics.max_drawdown_subunits ==
              maximum(p.drawdown_subunits for p in result.ledger)
        @test count(p -> p.kind == :venue_event, result.ledger) == length(events)
        @test count(p -> p.kind == :decision, result.ledger) == length(events)
        @test last(result.ledger).kind == :session_end
    end

    @testset "Repeated runs on one portfolio never alias fill ids" begin
        events = vcat(_base_events(), MarketEvent[_bd(3, 200, Bid, 100, 1)])
        decide = _at_sequence(2, OrderIntent(Buy, 2, 102; time_in_force = :ioc))
        portfolio = _portfolio()
        first_run = simulate!(portfolio, _session(events), decide; config = _config(10))
        second_run = simulate!(portfolio, _session(events), decide; config = _config(10))
        @test only(first_run.fills).fill_id == "r1-o1-f1"
        @test only(second_run.fills).fill_id == "r2-o1-f1"
        # Both runs are booked; nothing is swallowed as an idempotent replay.
        @test portfolio.inventory_lots == 4
        @test portfolio.cash_subunits == 100_000 - 2 * (2 * 102)
        @test count(row -> row.idempotent_replay, portfolio.audit) == 0
        @test portfolio.simulation_runs == 2
        @test second_run.assumptions["run_index"] == 2
        @test first_run.result_hash != second_run.result_hash
        # A run that fails is not counted and does not consume a run index.
        @test_throws ErrorException simulate!(
            portfolio,
            _session(events),
            _ -> error("x");
            config = _config(10),
        )
        @test portfolio.simulation_runs == 2
        # A stale fill id already in the portfolio is a conflict, never a silent replay.
        planted = _portfolio()
        apply_fill!(planted, "r1-o1-f1", 110, _SIM_INSTRUMENT, Buy, 102, 2, 0)
        cash = planted.cash_subunits
        @test_throws PortfolioError simulate!(
            planted,
            _session(events),
            decide;
            config = _config(10),
        )
        @test planted.cash_subunits == cash && planted.inventory_lots == 2
    end

    @testset "Reject before mutation" begin
        events = vcat(_base_events(), MarketEvent[_bd(3, 200, Bid, 100, 1)])
        limits = RiskLimits(
            max_order_lots = 5,
            max_inventory_lots = 6,
            max_initial_cash_subunits = 200_000,
            max_open_orders = 2,
        )
        cases = [
            (OrderIntent(Buy, 6, 102), "order_size_limit"),
            (OrderIntent(Sell, 1, 100), "unsupported_short"),
            (OrderIntent(Buy, 5, 102), "insufficient_cash"),
        ]
        for (intent, reason) in cases
            portfolio = _portfolio(cash = reason == "insufficient_cash" ? 500 : 100_000)
            cash = portfolio.cash_subunits
            result = simulate!(
                portfolio,
                _session(events),
                _at_sequence(2, intent);
                config = _config(5; limits = limits),
            )
            @test only(result.orders).status == :rejected
            @test only(result.decisions).reason == reason
            @test !only(result.decisions).accepted
            @test isempty(result.fills)
            @test portfolio.cash_subunits == cash && portfolio.inventory_lots == 0
            @test only(result.audit).kind == :reject
        end
        # Worst-case inventory includes live orders: 4 + 4 > 6.
        stacked = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(2, [OrderIntent(Buy, 4, 100), OrderIntent(Buy, 4, 100)]);
            config = _config(5; limits = limits),
        )
        @test [o.status for o in stacked.orders] == [:expired, :rejected]
        @test stacked.decisions[2].reason == "inventory_limit"
        # A post-only order that would take liquidity is rejected at activation, not repriced.
        post = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(2, OrderIntent(Buy, 1, 102; post_only = true));
            config = _config(5),
        )
        @test only(post.orders).status == :rejected && isempty(post.fills)
        # Bounded initial capital and transactional failure of the whole run.
        @test_throws ArgumentError simulate!(
            _portfolio(cash = 300_000),
            _session(events),
            _ -> nothing;
            config = _config(5; limits = limits),
        )
        portfolio = _portfolio()
        snapshot_before =
            (portfolio.cash_subunits, portfolio.inventory_lots, length(portfolio.audit))
        exploding =
            ctx ->
                ctx.info_sequence == 3 ? error("boom") :
                (ctx.info_sequence == 2 ? OrderIntent(Buy, 2, 102; time_in_force = :ioc) : nothing)
        @test_throws ErrorException simulate!(
            portfolio,
            _session(events),
            exploding;
            config = _config(5),
        )
        @test (portfolio.cash_subunits, portfolio.inventory_lots, length(portfolio.audit)) ==
              snapshot_before
        other = PaperPortfolio("ASSET_B", 1_000)
        @test_throws PortfolioError simulate!(
            other,
            _session(events),
            _ -> nothing;
            config = _config(5),
        )
        # Portfolio-level rejections leave state unchanged.
        p = _portfolio()
        @test_throws PortfolioError apply_fill!(p, "x", 1, _SIM_INSTRUMENT, Sell, 100, 1, 0)
        @test_throws PortfolioError apply_fill!(p, "x", 1, "ASSET_B", Buy, 100, 1, 0)
        @test_throws PortfolioError apply_fill!(
            p,
            "x",
            1,
            _SIM_INSTRUMENT,
            Buy,
            typemax(Int64),
            2,
            0,
        )
        @test (p.cash_subunits, p.inventory_lots, length(p.audit)) == (100_000, 0, 0)
    end

    @testset "Stale and missing prices are never invented" begin
        events = MarketEvent[
            _bd(1, 100, Bid, 100, 10),             # one-sided: no mid yet
            _bd(2, 110, Ask, 102, 10),             # first valid mid 101
            _bd(3, 200, Ask, 102, -10),            # ask side empty: mark is stale
            _bd(4, 300, Ask, 104, 10),             # fresh again, mid 102
        ]
        preloaded = _portfolio()
        apply_fill!(preloaded, "pre", 1, _SIM_INSTRUMENT, Buy, 100, 2, 0)
        result = simulate!(preloaded, _session(events), _ -> nothing; config = _config(0))
        venue_points = [p for p in result.ledger if p.kind == :venue_event]
        @test [p.mark_status for p in venue_points] == [:missing, :fresh, :stale, :fresh]
        @test venue_points[1].mark_price_ticks === nothing
        @test venue_points[1].unrealized_pnl_subunits === nothing
        @test venue_points[1].equity_subunits === nothing
        @test venue_points[3].mark_price_ticks == 101
        @test venue_points[4].mark_price_ticks == 102
        @test result.metrics.stale_mark_steps > 0 && result.metrics.missing_mark_steps > 0
        @test isempty(reconciliation_errors(result))
        # Feature rows are absent while the observed book is one-sided.
        seen = Any[]
        simulate!(
            _portfolio(),
            _session(events),
            ctx -> (push!(seen, ctx.features); nothing);
            config = _config(0),
        )
        @test seen[1] === nothing && seen[3] === nothing
        @test seen[2] isa FeatureRow && seen[2].mid_ticks == 101
    end

    @testset "Terminal open-position valuation" begin
        events =
            vcat(_base_events(), MarketEvent[_bd(3, 200, Bid, 101, 3), _bd(4, 300, Ask, 102, 1)])
        portfolio = _portfolio()
        result = simulate!(
            portfolio,
            _session(events),
            _at_sequence(2, OrderIntent(Buy, 3, 102; time_in_force = :ioc));
            config = _config(10),
        )
        m = result.metrics
        @test m.final_inventory_lots == 3
        @test m.final_mark_status == :fresh
        @test m.final_mark_price_ticks == fld(101 + 102, 2)
        @test m.final_unrealized_pnl_subunits == 101 * 3 - 102 * 3
        @test m.final_equity_subunits == portfolio.cash_subunits + 101 * 3
        @test m.realized_pnl_subunits == 0
    end

    @testset "Normalized decision and fill records" begin
        events = vcat(_base_events(), MarketEvent[_bd(3, 200, Bid, 100, 1)])
        result = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(2, OrderIntent(Buy, 2, 102; time_in_force = :ioc, tag = "probe"));
            config = _config(10),
        )
        decisions = decision_records(result)
        fills = fill_records(result)
        @test only(decisions)["record_type"] == "paper_decision"
        @test only(decisions)["paper_only"] === true
        @test only(decisions)["tag"] == "probe"
        @test !any(k -> occursin(r"pnl|outcome|label"i, k), keys(only(decisions)))
        @test only(fills)["record_type"] == "paper_fill"
        @test only(fills)["approximation"] === true
        @test only(fills)["decision_id"] == only(decisions)["decision_id"]
        roundtrip = JSON.parse(JSON.json(fills))
        @test roundtrip[1]["exchange_ts_ns"] == 110
    end

    @testset "metabolic-ledger parity.v1 golden fixtures (vendored)" begin
        provenance = JSON.parsefile(joinpath(_PARITY_DIR, "PROVENANCE.json"))
        @test provenance["schema"] == PARITY_SCHEMA_ID
        files = sort([
            f for f in readdir(_PARITY_DIR) if endswith(f, ".json") && f != "PROVENANCE.json"
        ])
        @test length(files) == 9
        @test Set(files) == Set(keys(provenance["files"]))
        for file in files
            path = joinpath(_PARITY_DIR, file)
            @test bytes2hex(open(sha256, path)) == provenance["files"][file]["sha256"]
            passed, failures, _ = run_parity_fixture(load_parity_fixture(path))
            @test passed
            isempty(failures) || @info "parity failure" file failures
        end
    end

    @testset "Simulator output round-trips through the parity.v1 contract" begin
        events = vcat(
            _base_events(),
            MarketEvent[
                _bd(3, 150, Bid, 101, 5),
                _tp(4, 200, Buy, 102, 3),
                _bd(5, 300, Bid, 100, 1),
            ],
        )
        actions = Dict{Int, Any}(
            2 => [OrderIntent(Buy, 7, 102; time_in_force = :ioc), OrderIntent(Sell, 50, 100)],
            3 => [OrderIntent(Sell, 3, 101; time_in_force = :ioc), OrderIntent(Sell, 2, 103)],
        )
        fees = FeeSchedule(maker_fee_ppm = -100, taker_fee_ppm = 300, per_fill_fee_subunits = 1)
        result = simulate!(
            _portfolio(),
            _session(events),
            _at_sequence(actions);
            config = _config(10; fees = fees),
        )
        @test result.in_parity_subset
        document = parity_document(result, "sim_roundtrip")
        @test document["source"] == "dendritetrader.jl"
        reparsed = JSON.parse(JSON.json(document))
        passed, failures, replayed = run_parity_fixture(reparsed)
        @test passed
        isempty(failures) || @info "roundtrip failures" failures
        @test replayed.cash_subunits == last(result.ledger).cash_subunits
        @test any(row -> row.kind == :reject, result.audit)
        @test any(row -> row.kind == :fill, result.audit)
    end

    @testset "Julia-only signed experiments stay outside the parity subset" begin
        events = vcat(
            _base_events(),
            MarketEvent[
                _bd(3, 150, Bid, 100, 1),
                _bd(4, 200, Ask, 102, 1),
                _bd(5, 300, Bid, 100, 1),
            ],
        )
        actions = Dict{Int, Any}(
            2 => OrderIntent(Buy, 4, 102; time_in_force = :ioc),
            3 => OrderIntent(Sell, 6, 100; time_in_force = :ioc),
        )
        long_flat =
            simulate!(_portfolio(), _session(events), _at_sequence(actions); config = _config(10))
        @test long_flat.decisions[2].reason == "unsupported_short"
        @test long_flat.metrics.final_inventory_lots == 4
        signed_portfolio = _portfolio(mode = :signed)
        signed = simulate!(
            signed_portfolio,
            _session(events),
            _at_sequence(actions);
            config = _config(10),
        )
        @test signed_portfolio.inventory_lots == -2
        @test !signed.in_parity_subset
        @test signed.assumptions["accounting"]["parity_subset"] == false
        @test_throws PortfolioError parity_document(signed, "signed")
        @test isempty(reconciliation_errors(signed))
        # Flip accounting: buy 4 @ 100, sell 6 @ 110 with fee 6 split 4 / 2 across legs.
        p = PaperPortfolio(_SIM_INSTRUMENT, 10_000; mode = :signed)
        apply_fill!(p, "a", 1, _SIM_INSTRUMENT, Buy, 100, 4, 0)
        apply_fill!(p, "b", 2, _SIM_INSTRUMENT, Sell, 110, 6, 6)
        @test p.inventory_lots == -2
        @test p.realized_pnl_subunits == 36
        @test p.total_cost_subunits == -218
        @test unrealized_pnl(p, 110) == -2
        @test p.fees_subunits == 6
        @test reconciles(p)
        # Short marks round half a tick against the position.
        short_book = MarketEvent[
            _bd(1, 100, Bid, 100, 10),
            _bd(2, 100, Ask, 103, 10),
            _bd(3, 200, Bid, 100, 1),
        ]
        sp = _portfolio(mode = :signed)
        short = simulate!(
            sp,
            _session(short_book),
            _at_sequence(2, OrderIntent(Sell, 1, 100; time_in_force = :ioc));
            config = _config(5),
        )
        @test last(short.ledger).mark_price_ticks == 102   # cld(203, 2)
    end
end
