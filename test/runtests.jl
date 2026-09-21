# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using JSON
using ZMQ
using DendriteTrader

@testset "DendriteTrader" begin
    @testset "TradeSignal" begin
        d = Dict(
            "ticker"=>"MARKET-A",
            "side"=>"BUY",
            "price"=>100.0,
            "quantity"=>1.0,
            "confidence"=>0.9,
            "timestamp_ns"=>0,
        )
        s = TradeSignal(d)
        @test s.ticker == "MARKET-A"
        @test s.side == Buy
        @test s.confidence ≈ 0.9f0
        @test passes_gate(s, 0.85f0)
        @test !passes_gate(s, 0.95f0)
    end

    @testset "latency_ns supports deterministic observation time" begin
        signal = TradeSignal("MARKET-A", Buy, 100.0, 1.0, 0.9f0, 1_000)
        @test latency_ns(signal, 1_750) == 750
        @test latency_ns(signal, 500) == 0
    end

    @testset "validate_signal" begin
        # Helper: valid signal dict
        valid = Dict(
            "ticker" => "MARKET-A",
            "side" => "BUY",
            "price" => 100.0,
            "quantity" => 1.0,
            "confidence" => 0.9,
            "timestamp_ns" => 1_000,
        )

        # Valid signal returns nothing
        @test validate_signal(valid) === nothing

        # Zero timestamp is rejected (must be > 0 per contract)
        zero_ts = copy(valid)
        zero_ts["timestamp_ns"] = 0
        @test occursin("timestamp", validate_signal(zero_ts))

        # Missing each required field returns error
        for field in ("ticker", "side", "price", "confidence", "timestamp_ns")
            d = copy(valid)
            delete!(d, field)
            r = validate_signal(d)
            @test r isa String
            @test occursin(field, r)
        end

        # Empty ticker returns error
        empty_ticker = copy(valid)
        empty_ticker["ticker"] = ""
        @test occursin("ticker", validate_signal(empty_ticker))

        # Invalid side returns error
        for bad_side in ("hold", "Buy", "", "BUY SELL")
            d = copy(valid)
            d["side"] = bad_side
            r = validate_signal(d)
            @test r isa String
            @test occursin("side", r)
        end

        # Negative price returns error
        neg_price = copy(valid)
        neg_price["price"] = -1.0
        @test occursin("price", validate_signal(neg_price))

        # Zero price returns error
        zero_price = copy(valid)
        zero_price["price"] = 0.0
        @test occursin("price", validate_signal(zero_price))

        # Confidence out of range returns error
        for bad_conf in (-0.1, 1.1, -1.0, 2.0)
            d = copy(valid)
            d["confidence"] = bad_conf
            @test occursin("confidence", validate_signal(d))
        end

        # Boundary confidence values are accepted
        for edge_conf in (0.0, 1.0)
            d = copy(valid)
            d["confidence"] = edge_conf
            @test validate_signal(d) === nothing
        end

        # Bool rejected for price
        bool_price = copy(valid)
        bool_price["price"] = true
        @test occursin("number", validate_signal(bool_price))

        # Bool rejected for confidence
        bool_conf = copy(valid)
        bool_conf["confidence"] = false
        @test occursin("number", validate_signal(bool_conf))

        # Bool rejected for timestamp
        bool_ts = copy(valid)
        bool_ts["timestamp_ns"] = true
        @test occursin("integer", validate_signal(bool_ts))

        # NaN rejected for price
        nan_price = copy(valid)
        nan_price["price"] = NaN
        @test occursin("finite", validate_signal(nan_price))

        # Inf rejected for price
        inf_price = copy(valid)
        inf_price["price"] = Inf
        @test occursin("finite", validate_signal(inf_price))

        # NaN rejected for confidence
        nan_conf = copy(valid)
        nan_conf["confidence"] = NaN
        @test occursin("finite", validate_signal(nan_conf))

        # Inf rejected for confidence
        inf_conf = copy(valid)
        inf_conf["confidence"] = Inf
        @test occursin("finite", validate_signal(inf_conf))

        # Negative timestamp returns error
        neg_ts = copy(valid)
        neg_ts["timestamp_ns"] = -1
        @test occursin("timestamp", validate_signal(neg_ts))
    end

    @testset "ExecutionEngine — confidence gate" begin
        engine = ExecutionEngine(confidence_threshold = 0.85f0)

        # Signal above threshold -> executed
        sig_hi = TradeSignal(
            Dict(
                "ticker"=>"MARKET-B",
                "side"=>"BUY",
                "price"=>90.0,
                "quantity"=>1.0,
                "confidence"=>0.90,
                "timestamp_ns"=>0,
            ),
        )
        dec = execute_signal!(engine, sig_hi, 10_000.0)
        @test dec.executed
        @test dec.position_units > 0.0
        @test dec.kelly_fraction > 0.0
        @test dec.applied_fraction > 0.0

        # Signal below threshold -> rejected
        sig_lo = TradeSignal(
            Dict(
                "ticker"=>"MARKET-B",
                "side"=>"BUY",
                "price"=>90.0,
                "quantity"=>1.0,
                "confidence"=>0.70,
                "timestamp_ns"=>0,
            ),
        )
        dec_lo = execute_signal!(engine, sig_lo, 10_000.0)
        @test !dec_lo.executed
    end

    @testset "ExecutionEngine — position tracking" begin
        engine = ExecutionEngine()
        sig = TradeSignal(
            Dict(
                "ticker"=>"MARKET-C",
                "side"=>"BUY",
                "price"=>0.03,
                "quantity"=>100.0,
                "confidence"=>0.92,
                "timestamp_ns"=>0,
            ),
        )
        execute_signal!(engine, sig, 500.0)
        @test get(engine.positions, "MARKET-C", 0.0) > 0.0
    end

    @testset "fill_rate" begin
        engine = ExecutionEngine(confidence_threshold = 0.85f0)
        for conf in [0.90, 0.70, 0.88, 0.60]
            s = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-A",
                    "side"=>"BUY",
                    "price"=>100.0,
                    "quantity"=>1.0,
                    "confidence"=>conf,
                    "timestamp_ns"=>0,
                ),
            )
            execute_signal!(engine, s, 10_000.0)
        end
        @test 0.0 < fill_rate(engine) < 1.0
    end

    @testset "DydxPrice" begin
        p = DydxPrice("MARKET-A", 100.0, 99.0, 101.0)
        @test mid_price(p) ≈ 100.0
        @test spread_bps(p) > 0.0
    end

    @testset "Kelly sizing" begin
        f = kelly_fraction(win_rate = 0.55, avg_win = 8.50, avg_loss = 5.20)
        @test 0.02 <= f <= 0.20
        @test f > 0.05

        f_bad = kelly_fraction(win_rate = 0.40, avg_win = 1.0, avg_loss = 2.0)
        @test f_bad == 0.0

        f_zero = kelly_fraction(win_rate = 0.0, avg_win = 1_000, avg_loss = 1)
        @test f_zero == 0.0

        f_hi = kelly_fraction(win_rate = 0.70, avg_win = 10.0, avg_loss = 5.0)
        f_lo = kelly_fraction(win_rate = 0.52, avg_win = 10.0, avg_loss = 5.0)
        @test f_hi > f_lo

        f_conf_high = from_confidence(confidence = 0.95)
        f_conf_low = from_confidence(confidence = 0.55)
        @test f_conf_high > f_conf_low
        @test 0.0 <= f_conf_low <= 1.0
        @test 0.0 <= f_conf_high <= 1.0

        hk = half_kelly(0.60, 1.5)
        @test 0.0 <= hk <= 1.0
        p, b = 0.60, 1.5
        q = 1.0 - p
        full = (p * b - q) / b
        @test half_kelly(p, b) ≈ full * 0.5 atol=1e-9

        @test risk_tier(0.97) == Aggressive
        @test risk_tier(0.90) == Moderate
        @test risk_tier(0.75) == Conservative
        @test risk_tier(0.50) == Minimal

        pos = size_position(confidence = 0.90, price = 100.0, account_balance = 10_000.0)
        @test pos.units > 0.0
        @test pos.kelly_fraction > 0.0
        @test pos.risk == Moderate
        @test 0.0 < pos.account_risk_pct < 100.0

        pos_from_ints = size_position(confidence = 0.90f0, price = 100, account_balance = 10_000)
        @test pos_from_ints.units > 0.0

        pos_hi = size_position(confidence = 0.95, price = 100.0, account_balance = 10_000.0)
        pos_lo = size_position(confidence = 0.72, price = 100.0, account_balance = 10_000.0)
        @test pos_hi.units >= pos_lo.units

        @test size_position(confidence = 0.90, price = 0.0, account_balance = 10_000.0).units == 0.0
        @test size_position(confidence = 0.90, price = 100.0, account_balance = -1.0).units == 0.0
    end

    @testset "ExecutionEngine — zero-sized rejection and capped units" begin
        engine = ExecutionEngine(max_position_size = 10.0)
        sig = TradeSignal(
            Dict(
                "ticker"=>"MARKET-D",
                "side"=>"BUY",
                "price"=>90.0,
                "quantity"=>1.0,
                "confidence"=>0.90,
                "timestamp_ns"=>0,
            ),
        )
        dec = execute_signal!(engine, sig, 10_000.0)
        @test dec.executed
        @test dec.position_units == 10.0
        k_model = size_position(
            confidence = Float64(sig.confidence),
            price = 90.0,
            account_balance = 10_000.0,
            payoff_ratio = engine.payoff_ratio,
        ).kelly_fraction
        @test dec.kelly_fraction ≈ k_model
        @test dec.kelly_fraction > dec.applied_fraction
        @test dec.applied_fraction ≈ 0.09

        zero_dec = execute_signal!(engine, sig, 0.0)
        @test !zero_dec.executed
        @test zero_dec.reason == "zero-sized position"
    end

    @testset "PriceCache" begin
        @testset "constructor with default TTL" begin
            cache = PriceCache()
            @test cache.ttl_s == 5.0
            @test cache_size(cache) == 0
        end

        @testset "constructor with custom TTL" begin
            cache = PriceCache(ttl_s = 10.0)
            @test cache.ttl_s == 10.0
        end

        @testset "put_cached! stores entry" begin
            cache = PriceCache()
            price = DydxPrice("BTC-USD", 50000.0, 49999.0, 50001.0)
            put_cached!(cache, "BTC-USD", price)
            @test cache_size(cache) == 1
        end

        @testset "get_cached returns fresh entry" begin
            cache = PriceCache()
            price = DydxPrice("ETH-USD", 3000.0, 2999.0, 3001.0)
            put_cached!(cache, "ETH-USD", price)
            result = get_cached(cache, "ETH-USD")
            @test result !== nothing
            @test result.ticker == "ETH-USD"
            @test result.oracle_price ≈ 3000.0
        end

        @testset "get_cached returns nothing for expired entry" begin
            cache = PriceCache(ttl_s = 5.0)
            price = DydxPrice("SOL-USD", 100.0, 99.0, 101.0)
            put_cached!(cache, "SOL-USD", price)
            # Deterministic expiry: backdate the stored timestamp (no sleep)
            lock(cache.lock) do
                cache.times["SOL-USD"] = time() - 10.0
            end
            @test get_cached(cache, "SOL-USD") === nothing
            # get_cached lazily evicts expired entries
            @test cache_size(cache) == 0
        end

        @testset "get_cached returns nothing for missing ticker" begin
            cache = PriceCache()
            @test get_cached(cache, "MISSING") === nothing
        end

        @testset "invalidate! removes specific entry" begin
            cache = PriceCache()
            put_cached!(cache, "BTC-USD", DydxPrice("BTC-USD", 50000.0, 49999.0, 50001.0))
            put_cached!(cache, "ETH-USD", DydxPrice("ETH-USD", 3000.0, 2999.0, 3001.0))
            @test cache_size(cache) == 2
            invalidate!(cache, "BTC-USD")
            @test cache_size(cache) == 1
            @test get_cached(cache, "BTC-USD") === nothing
            @test get_cached(cache, "ETH-USD") !== nothing
        end

        @testset "invalidate! on missing ticker is a no-op" begin
            cache = PriceCache()
            invalidate!(cache, "MISSING")
            @test cache_size(cache) == 0
        end

        @testset "clear! removes all entries" begin
            cache = PriceCache()
            put_cached!(cache, "BTC-USD", DydxPrice("BTC-USD", 50000.0, 49999.0, 50001.0))
            put_cached!(cache, "ETH-USD", DydxPrice("ETH-USD", 3000.0, 2999.0, 3001.0))
            put_cached!(cache, "SOL-USD", DydxPrice("SOL-USD", 100.0, 99.0, 101.0))
            @test cache_size(cache) == 3
            clear!(cache)
            @test cache_size(cache) == 0
            @test get_cached(cache, "BTC-USD") === nothing
            @test get_cached(cache, "ETH-USD") === nothing
            @test get_cached(cache, "SOL-USD") === nothing
        end

        @testset "clear! on empty cache is a no-op" begin
            cache = PriceCache()
            clear!(cache)
            @test cache_size(cache) == 0
        end

        @testset "cache_size returns correct count" begin
            cache = PriceCache()
            @test cache_size(cache) == 0
            put_cached!(cache, "A", DydxPrice("A", 1.0, 0.9, 1.1))
            @test cache_size(cache) == 1
            put_cached!(cache, "B", DydxPrice("B", 2.0, 1.9, 2.1))
            @test cache_size(cache) == 2
            put_cached!(cache, "A", DydxPrice("A", 1.5, 1.4, 1.6))
            @test cache_size(cache) == 2
        end
    end
    @testset "Sell and Neutral signals" begin
        @testset "Sell signal on existing position decrements position" begin
            engine = ExecutionEngine()
            buy = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-E",
                    "side"=>"BUY",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            dec_buy = execute_signal!(engine, buy, 10_000.0)
            @test dec_buy.executed
            pos_after_buy = engine.positions["MARKET-E"]
            @test pos_after_buy > 0.0

            sell = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-E",
                    "side"=>"SELL",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            dec_sell = execute_signal!(engine, sell, 10_000.0)
            @test dec_sell.executed
            @test engine.positions["MARKET-E"] < pos_after_buy
        end

        @testset "Sell signal on empty position opens a signed short" begin
            engine = ExecutionEngine()
            sell = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-F",
                    "side"=>"SELL",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            dec = execute_signal!(engine, sell, 10_000.0)
            @test dec.executed
            @test engine.positions["MARKET-F"] == -dec.position_units
        end

        @testset "Neutral signal rejected with 'neutral signal' reason" begin
            engine = ExecutionEngine()
            neutral = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-G",
                    "side"=>"NEUTRAL",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            dec = execute_signal!(engine, neutral, 10_000.0)
            @test !dec.executed
            @test dec.reason == "neutral signal"
        end

        @testset "Buy then Sell — position goes to zero" begin
            engine = ExecutionEngine()
            buy = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-H",
                    "side"=>"BUY",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            execute_signal!(engine, buy, 10_000.0)
            @test engine.positions["MARKET-H"] > 0.0

            sell = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-H",
                    "side"=>"SELL",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            execute_signal!(engine, sell, 10_000.0)
            @test engine.positions["MARKET-H"] == 0.0
        end

        @testset "Multiple Buy then Sell — correct netting" begin
            engine = ExecutionEngine()
            buy_sig = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-I",
                    "side"=>"BUY",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            execute_signal!(engine, buy_sig, 10_000.0)
            execute_signal!(engine, buy_sig, 10_000.0)
            pos_after_two_buys = engine.positions["MARKET-I"]

            sell_sig = TradeSignal(
                Dict(
                    "ticker"=>"MARKET-I",
                    "side"=>"SELL",
                    "price"=>50.0,
                    "quantity"=>1.0,
                    "confidence"=>0.92,
                    "timestamp_ns"=>1_000,
                ),
            )
            execute_signal!(engine, sell_sig, 10_000.0)
            # Default max_position_size=10; two capped buys then one sell → exactly 10.0
            @test engine.positions["MARKET-I"] == 10.0
            @test engine.positions["MARKET-I"] < pos_after_two_buys
        end
    end
    @testset "Backtest" begin
        @testset "BacktestConfig constructor with defaults" begin
            cfg = BacktestConfig()
            @test cfg.initial_balance == 10_000.0
            @test cfg.confidence_threshold == Float32(0.85)
            @test cfg.payoff_ratio == 1.5
            @test cfg.max_position_size == 10.0
            @test cfg.slippage_pct == 0.0
            @test cfg.commission_pct == 0.0
        end

        @testset "BacktestConfig constructor with custom values" begin
            cfg = BacktestConfig(
                initial_balance = 50_000.0,
                confidence_threshold = Float32(0.90),
                payoff_ratio = 2.0,
                max_position_size = 20.0,
                slippage_pct = 0.05,
                commission_pct = 0.1,
            )
            @test cfg.initial_balance == 50_000.0
            @test cfg.confidence_threshold == Float32(0.90)
            @test cfg.payoff_ratio == 2.0
            @test cfg.max_position_size == 20.0
            @test cfg.slippage_pct == 0.05
            @test cfg.commission_pct == 0.1
        end

        @testset "run_backtest with buy signals — equity curve changes" begin
            cfg = BacktestConfig(initial_balance = 10_000.0)
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
                TradeSignal(
                    Dict(
                        "ticker" => "ETH-USD",
                        "side" => "BUY",
                        "price" => 50.0,
                        "quantity" => 1.0,
                        "confidence" => 0.90,
                        "timestamp_ns" => 2_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            @test length(result.equity_curve) == 3  # initial + 2 signals
            @test result.equity_curve[1] == 10_000.0
            # Buy-only: no trades recorded until a sell closes the position
            @test result.total_trades == 0
            @test result.final_balance == cfg.initial_balance
        end

        @testset "run_backtest with buy+sell — PnL computed" begin
            # max_position_size default 10.0 caps Kelly units → deterministic PnL
            cfg = BacktestConfig(initial_balance = 10_000.0)
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "SELL",
                        "price" => 110.0,
                        "quantity" => 1.0,
                        "confidence" => 0.90,
                        "timestamp_ns" => 2_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            @test result.final_balance != cfg.initial_balance
            @test result.total_return != 0.0
            closed = filter(t -> t.pnl != 0.0, result.trade_log)
            @test length(closed) >= 1
            @test closed[1].pnl > 0.0  # bought at 100, sold at 110
        end

        @testset "run_backtest with neutral signals — no trades" begin
            cfg = BacktestConfig(initial_balance = 10_000.0)
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "NEUTRAL",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            @test result.total_trades == 0
            @test result.final_balance == cfg.initial_balance
            @test result.equity_curve == [10_000.0, 10_000.0]
        end

        @testset "run_backtest with slippage — execution price adjusted" begin
            cfg = BacktestConfig(
                initial_balance = 10_000.0,
                slippage_pct = 0.5,  # 0.5% slippage
            )
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "SELL",
                        "price" => 110.0,
                        "quantity" => 1.0,
                        "confidence" => 0.90,
                        "timestamp_ns" => 2_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            # With 0.5% slippage, sell price = 110 * (1 - 0.5/100) = 109.45
            # Buy price stored as 100 * (1 + 0.5/100) = 100.5
            # PnL = (109.45 - 100.5) * units — should be positive but less than without slippage
            @test result.total_trades == 1
            @test result.trade_log[1].price ≈ 109.45 atol=0.01
            # Final balance should differ from no-slippage case
            cfg_noslip = BacktestConfig(initial_balance = 10_000.0)
            result_noslip = run_backtest(cfg_noslip, signals)
            @test result.final_balance < result_noslip.final_balance
        end

        @testset "run_backtest with commission — balance reduced" begin
            cfg = BacktestConfig(
                initial_balance = 10_000.0,
                commission_pct = 0.1,  # 0.1% commission
            )
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            # Commission should reduce balance: units * price * 0.1%
            @test result.final_balance < cfg.initial_balance
        end

        @testset "equity curve marks open positions at every signal" begin
            cfg = BacktestConfig(initial_balance = 10_000.0, max_position_size = 10.0)
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "NEUTRAL",
                        "price" => 110.0,
                        "quantity" => 0.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 2_000_000_000,
                    ),
                ),
            ]

            result = run_backtest(cfg, signals)

            @test result.equity_curve == [10_000.0, 10_000.0, 10_100.0]
            @test result.final_balance == last(result.equity_curve)
            @test result.total_return == 1.0
        end

        @testset "same-side observations update open-position marks" begin
            cfg = BacktestConfig(initial_balance = 10_000.0, max_position_size = 10.0)
            signals = [
                TradeSignal("BTC-USD", Buy, 100.0, 1.0, 0.92f0, 1_000_000_000),
                TradeSignal("BTC-USD", Buy, 90.0, 1.0, 0.92f0, 2_000_000_000),
            ]

            result = run_backtest(cfg, signals)

            @test result.equity_curve == [10_000.0, 10_000.0, 9_900.0]
            @test result.max_drawdown == 1.0
        end

        @testset "replay event latency is deterministic" begin
            cfg = BacktestConfig(initial_balance = 10_000.0)
            signals = [
                TradeSignal("BTC-USD", Buy, 100.0, 1.0, 0.92f0, 1_000_000_000),
                TradeSignal("BTC-USD", Sell, 101.0, 1.0, 0.90f0, 2_000_000_000),
            ]

            first = run_backtest(cfg, signals)
            second = run_backtest(cfg, signals)

            @test getfield.(first.events, :latency_ns) == [0, 0]
            @test getfield.(first.events, :latency_ns) == getfield.(second.events, :latency_ns)
        end

        @testset "full closes use one fill quantity across engine and ledger" begin
            cfg = BacktestConfig(initial_balance = 10_000.0, max_position_size = 1_000.0)
            long_signals = [
                TradeSignal("BTC-USD", Buy, 100.0, 1.0, 0.99f0, 1_000_000_000),
                TradeSignal("BTC-USD", Sell, 110.0, 1.0, 0.86f0, 2_000_000_000),
            ]
            short_signals = [
                TradeSignal("BTC-USD", Sell, 100.0, 1.0, 0.99f0, 1_000_000_000),
                TradeSignal("BTC-USD", Buy, 90.0, 1.0, 0.86f0, 2_000_000_000),
            ]

            long_result = run_backtest(cfg, long_signals)
            short_result = run_backtest(cfg, short_signals)

            @test long_result.trade_log[end].is_closed
            @test long_result.events[end].position_units == long_result.trade_log[end].units
            @test short_result.trade_log[end].is_closed
            @test short_result.events[end].position_units == short_result.trade_log[end].units
        end

        @testset "run_backtest with slippage and commission combined" begin
            cfg =
                BacktestConfig(initial_balance = 10_000.0, slippage_pct = 0.5, commission_pct = 0.1)
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "SELL",
                        "price" => 110.0,
                        "quantity" => 1.0,
                        "confidence" => 0.90,
                        "timestamp_ns" => 2_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            # trade_log has the sell trade only (buy-only doesn't create trade records)
            # Sell with 0.5% slippage: 110 * (1 - 0.5/100) = 109.45
            @test result.trade_log[1].price ≈ 109.45 atol=0.01
            # Commission applied on both buy and sell, so balance should be reduced
            @test result.final_balance != cfg.initial_balance
        end

        @testset "print_summary — no errors" begin
            cfg = BacktestConfig()
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            @test_nowarn print_summary(result)
        end

        @testset "format_currency" begin
            fmt = DendriteTrader.Backtest.format_currency
            @test fmt(1234.56) == "1,234.56"
            @test fmt(-5000.10) == "-5,000.10"
            @test fmt(0.0) == "0.00"
            @test fmt(1_000_000.00) == "1,000,000.00"
            @test fmt(NaN) == "NaN"
            @test fmt(Inf) == "Inf"
            @test fmt(-Inf) == "-Inf"
        end

        @testset "compute_max_drawdown" begin
            mdd = DendriteTrader.Backtest.compute_max_drawdown
            # Flat curve → no drawdown
            @test mdd([100.0, 100.0, 100.0]) == 0.0
            # Single element → 0
            @test mdd([100.0]) == 0.0
            # Dip from 100 to 80 → 20% drawdown
            @test mdd([100.0, 110.0, 88.0]) ≈ 20.0 atol=0.01
            # Recover after dip
            @test mdd([100.0, 120.0, 90.0, 130.0]) ≈ 25.0 atol=0.01
        end

        @testset "close PnL uses open units not exit Kelly size" begin
            cfg = BacktestConfig(initial_balance = 10_000.0, max_position_size = 1000.0)
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "BUY",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.99,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "SELL",
                        "price" => 110.0,
                        "quantity" => 1.0,
                        "confidence" => 0.86,
                        "timestamp_ns" => 2_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            closed = filter(t -> t.pnl != 0.0, result.trade_log)
            @test length(closed) == 1
            eng = ExecutionEngine(
                confidence_threshold = Float32(0.85),
                max_position_size = 1000.0,
                payoff_ratio = 1.5,
            )
            open_dec = execute_signal!(eng, signals[1], 10_000.0)
            @test closed[1].units ≈ open_dec.position_units atol=1e-9
            @test closed[1].pnl ≈ (110.0 - 100.0) * open_dec.position_units atol=1e-6
        end

        @testset "integer kwargs coerce into BacktestConfig" begin
            cfg = BacktestConfig(initial_balance = 10_000, slippage_pct = 0, commission_pct = 0)
            @test cfg.initial_balance == 10_000.0
            @test cfg.slippage_pct == 0.0
        end

        @testset "short open sell slippage fills below mid" begin
            cfg = BacktestConfig(initial_balance = 10_000.0, slippage_pct = 1.0)
            signals = [
                TradeSignal(
                    Dict(
                        "ticker" => "BTC-USD",
                        "side" => "SELL",
                        "price" => 100.0,
                        "quantity" => 1.0,
                        "confidence" => 0.92,
                        "timestamp_ns" => 1_000_000_000,
                    ),
                ),
            ]
            result = run_backtest(cfg, signals)
            @test result.total_trades == 1
            @test result.trade_log[1].price ≈ 99.0 atol=1e-9
        end

        @testset "compute_sharpe_ratio with per-signal risk-free" begin
            sharpe = DendriteTrader.Backtest.compute_sharpe_ratio
            r = [0.01, 0.02, -0.005, 0.015, 0.008]
            s = sharpe(r, 0.0, 5, false)
            @test s > 0.0
            @test isfinite(s)
            s_low = sharpe(r, 0.005, 5, false)
            @test s_low < s
        end

        @testset "compute_sortino_ratio consistent with Sharpe" begin
            sortino = DendriteTrader.Backtest.compute_sortino_ratio
            sharpe = DendriteTrader.Backtest.compute_sharpe_ratio
            r = [0.02, -0.01, 0.03, -0.005, 0.01]
            s = sortino(r, 0.0, 5, false)
            @test isfinite(s)
            @test s > 0.0
            @test s >= sharpe(r, 0.0, 5, false) - 1e-9
        end
    end
end

@testset "strip_zmq_topic handles ASCII and multi-byte UTF-8" begin
    strip = DendriteTrader.strip_zmq_topic
    @test strip("trade.{\"a\":1}", "trade.") == "{\"a\":1}"
    # Multi-byte prefix: "éa" is 3 bytes (é=2, a=1), 2 chars
    topic = "éa"
    payload = "{\"x\":1}"
    msg = topic * payload
    @test startswith(msg, topic)
    @test strip(msg, topic) == payload
    @test strip("no-prefix", "trade.") == "no-prefix"
    @test strip("trade.", "trade.") == ""
end

@testset "ZMQ topic filtering" begin
    endpoint = "ipc:///tmp/dendrite-test-topic-$(getpid())-$(time_ns()).ipc"
    ctx = ZMQ.Context()
    pub = ZMQ.Socket(ctx, ZMQ.PUB)
    ZMQ.bind(pub, endpoint)
    sleep(0.1)

    engine = ExecutionEngine(confidence_threshold = Float32(0.50))
    decisions = ExecutionDecision[]

    t = @async start!(
        engine,
        zmq_endpoint = endpoint,
        zmq_topic = "trade.",
        account_balance = 10_000.0,
        on_decision = d -> push!(decisions, d),
        timeout_s = 5.0,
    )
    # Allow SUB to connect and subscription to propagate (slow joiner)
    sleep(0.5)

    signal_json = JSON.json(
        Dict(
            "ticker" => "BTC-USD",
            "side" => "BUY",
            "price" => 50_000.0,
            "quantity" => 0.1,
            "confidence" => 0.92,
            "timestamp_ns" => 1_000_000_000,
        ),
    )

    # Heartbeat should be filtered out by topic subscription
    ZMQ.send(pub, "heartbeat." * JSON.json(Dict("status" => "ok")))

    # Resend until both trade messages are observed (or deadline)
    deadline = time() + 4.0
    while length(decisions) < 2 && time() < deadline
        ZMQ.send(pub, "trade." * signal_json)
        sleep(0.2)
    end

    # Assert topic filtering: both trade.* messages processed, heartbeat.* ignored
    @test length(decisions) == 2

    stop!(engine)
    wait(t)
    close(pub)
    close(ctx)
end

@testset "RateLimiter" begin
    @testset "constructor defaults" begin
        rl = RateLimiter()
        @test rl.tokens == 10.0
        @test rl.max_tokens == 10.0
        @test rl.refill_rate == 10.0
    end

    @testset "constructor rejects requests_per_second <= 0" begin
        @test_throws ArgumentError RateLimiter(requests_per_second = 0.0)
        @test_throws ArgumentError RateLimiter(requests_per_second = -1.0)
    end

    @testset "constructor rejects burst <= 0" begin
        @test_throws ArgumentError RateLimiter(burst = 0.0)
        @test_throws ArgumentError RateLimiter(burst = -5.0)
    end

    @testset "acquire! with available tokens" begin
        rl = RateLimiter(requests_per_second = 100.0, burst = 5.0)
        @test rl.tokens == 5.0
        acquire!(rl)
        @test rl.tokens ≈ 4.0 atol = 0.01
    end

    @testset "set_rate! changes refill rate" begin
        rl = RateLimiter(requests_per_second = 10.0, burst = 10.0)
        set_rate!(rl, 20.0)
        @test rl.refill_rate == 20.0
    end

    @testset "set_rate! preserves burst capacity" begin
        rl = RateLimiter(requests_per_second = 10.0, burst = 5.0)
        set_rate!(rl, 20.0)
        @test rl.max_tokens == 5.0
    end
end

@testset "PriceCache" begin
    @testset "constructor with default TTL" begin
        cache = PriceCache()
        @test cache.ttl_s == 5.0
        @test cache_size(cache) == 0
    end

    @testset "constructor with custom TTL" begin
        cache = PriceCache(ttl_s = 10.0)
        @test cache.ttl_s == 10.0
    end

    @testset "put_cached! stores entry" begin
        cache = PriceCache()
        price = DydxPrice("BTC-USD", 50000.0, 49999.0, 50001.0)
        put_cached!(cache, "BTC-USD", price)
        @test cache_size(cache) == 1
    end

    @testset "get_cached returns fresh entry" begin
        cache = PriceCache()
        price = DydxPrice("ETH-USD", 3000.0, 2999.0, 3001.0)
        put_cached!(cache, "ETH-USD", price)
        result = get_cached(cache, "ETH-USD")
        @test result !== nothing
        @test result.ticker == "ETH-USD"
        @test result.oracle_price ≈ 3000.0
    end

    @testset "get_cached returns nothing for expired entry" begin
        cache = PriceCache(ttl_s = 0.01)
        price = DydxPrice("SOL-USD", 100.0, 99.0, 101.0)
        put_cached!(cache, "SOL-USD", price)
        lock(cache.lock) do
            cache.times["SOL-USD"] = time() - cache.ttl_s - 1.0
        end
        @test get_cached(cache, "SOL-USD") === nothing
    end

    @testset "get_cached returns nothing for missing ticker" begin
        cache = PriceCache()
        @test get_cached(cache, "MISSING") === nothing
    end

    @testset "invalidate! removes specific entry" begin
        cache = PriceCache()
        put_cached!(cache, "BTC-USD", DydxPrice("BTC-USD", 50000.0, 49999.0, 50001.0))
        put_cached!(cache, "ETH-USD", DydxPrice("ETH-USD", 3000.0, 2999.0, 3001.0))
        @test cache_size(cache) == 2
        invalidate!(cache, "BTC-USD")
        @test cache_size(cache) == 1
        @test get_cached(cache, "BTC-USD") === nothing
        @test get_cached(cache, "ETH-USD") !== nothing
    end

    @testset "invalidate! on missing ticker is a no-op" begin
        cache = PriceCache()
        invalidate!(cache, "MISSING")
        @test cache_size(cache) == 0
    end

    @testset "clear! removes all entries" begin
        cache = PriceCache()
        put_cached!(cache, "BTC-USD", DydxPrice("BTC-USD", 50000.0, 49999.0, 50001.0))
        put_cached!(cache, "ETH-USD", DydxPrice("ETH-USD", 3000.0, 2999.0, 3001.0))
        put_cached!(cache, "SOL-USD", DydxPrice("SOL-USD", 100.0, 99.0, 101.0))
        @test cache_size(cache) == 3
        clear!(cache)
        @test cache_size(cache) == 0
        @test get_cached(cache, "BTC-USD") === nothing
        @test get_cached(cache, "ETH-USD") === nothing
        @test get_cached(cache, "SOL-USD") === nothing
    end

    @testset "clear! on empty cache is a no-op" begin
        cache = PriceCache()
        clear!(cache)
        @test cache_size(cache) == 0
    end

    @testset "cache_size returns correct count" begin
        cache = PriceCache()
        @test cache_size(cache) == 0
        put_cached!(cache, "A", DydxPrice("A", 1.0, 0.9, 1.1))
        @test cache_size(cache) == 1
        put_cached!(cache, "B", DydxPrice("B", 2.0, 1.9, 2.1))
        @test cache_size(cache) == 2
        put_cached!(cache, "A", DydxPrice("A", 1.5, 1.4, 1.6))
        @test cache_size(cache) == 2
    end
end

@testset "ExecutionEngine event persistence" begin
    mktempdir() do dir
        path = joinpath(dir, "events.jsonl")

        # Default engine has no log file
        default_engine = ExecutionEngine()
        @test default_engine.log_file === nothing
        @test default_engine.log_io === nothing

        sig = TradeSignal(
            Dict(
                "ticker" => "BTC-USD",
                "side" => "BUY",
                "price" => 100.0,
                "quantity" => 1.0,
                "confidence" => 0.92,
                "timestamp_ns" => 1_000_000_000,
            ),
        )
        execute_signal!(default_engine, sig, 10_000.0)
        @test !isfile(path)

        engine = ExecutionEngine(log_file = path)
        execute_signal!(engine, sig, 10_000.0)

        # Rejected signal should also be persisted
        sig_low = TradeSignal(
            Dict(
                "ticker" => "BTC-USD",
                "side" => "BUY",
                "price" => 100.0,
                "quantity" => 1.0,
                "confidence" => 0.50,
                "timestamp_ns" => 2_000_000_000,
            ),
        )
        execute_signal!(engine, sig_low, 10_000.0)

        close_log!(engine)

        @test isfile(path)
        loaded = load_history(path)
        @test length(loaded) == 2
        @test loaded[1].event_type == "executed"
        @test loaded[1].executed
        @test loaded[1].position_units > 0.0
        @test loaded[1].applied_fraction > 0.0
        @test loaded[2].event_type == "gate_reject"
        @test !loaded[2].executed
        @test loaded[2].reason == "confidence=0.5 < threshold=0.85"

        # Round-trip equality against in-memory events
        for (orig, ld) in zip(events(engine), loaded)
            @test orig.event_type == ld.event_type
            @test orig.ticker == ld.ticker
            @test orig.confidence ≈ ld.confidence
            @test orig.side == ld.side
            @test orig.reason == ld.reason
            @test orig.kelly_fraction ≈ ld.kelly_fraction
            @test orig.latency_ns == ld.latency_ns
            @test orig.timestamp ≈ ld.timestamp atol = 1e-6
            @test orig.executed == ld.executed
            @test orig.position_units ≈ ld.position_units
            @test orig.applied_fraction ≈ ld.applied_fraction
        end

        # Truncate clears an existing log
        engine2 = ExecutionEngine(log_file = path, truncate = true)
        close_log!(engine2)
        @test isfile(path)
        @test filesize(path) == 0
    end
end

@testset "load_history resilience" begin
    mktempdir() do dir
        missing_path = joinpath(dir, "missing.jsonl")
        @test load_history(missing_path) == SignalEvent[]

        empty_path = joinpath(dir, "empty.jsonl")
        touch(empty_path)
        @test load_history(empty_path) == SignalEvent[]

        bad_path = joinpath(dir, "bad.jsonl")
        write(
            bad_path,
            "not json\n\n" *
            "{\"schema_version\":1,\"event_type\":\"x\",\"ticker\":\"T\",\"confidence\":0.5,\"side\":\"BUY\"}\n" *
            "{\"schema_version\":99,\"event_type\":\"y\"}\n",
        )
        loaded = load_history(bad_path)
        @test length(loaded) == 1
        @test loaded[1].event_type == "x"
        @test loaded[1].ticker == "T"
    end
end

@testset "Backtest log_file" begin
    mktempdir() do dir
        path = joinpath(dir, "backtest.jsonl")
        cfg = BacktestConfig(initial_balance = 10_000.0, log_file = path)
        signals = [
            TradeSignal(
                Dict(
                    "ticker" => "BTC-USD",
                    "side" => "BUY",
                    "price" => 100.0,
                    "quantity" => 1.0,
                    "confidence" => 0.92,
                    "timestamp_ns" => 1_000_000_000,
                ),
            ),
            TradeSignal(
                Dict(
                    "ticker" => "BTC-USD",
                    "side" => "SELL",
                    "price" => 110.0,
                    "quantity" => 1.0,
                    "confidence" => 0.90,
                    "timestamp_ns" => 2_000_000_000,
                ),
            ),
        ]
        result = run_backtest(cfg, signals)
        @test length(result.events) == 2  # two executed signals

        loaded = load_history(path)
        @test length(loaded) == 2
        @test loaded[1].event_type == "executed"
        @test loaded[1].executed
        @test loaded[2].event_type == "executed"
        @test loaded[2].executed

        # run_backtest log_file kwarg overrides BacktestConfig
        override_path = joinpath(dir, "override.jsonl")
        run_backtest(cfg, signals; log_file = override_path)
        @test isfile(override_path)
        @test length(load_history(override_path)) == 2
    end
end

# Integration tests (gated behind DYDX_INTEGRATION=true)
include("integration/test_dydx.jl")
