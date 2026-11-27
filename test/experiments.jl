# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using JSON
using SHA
using TOML
using DendriteTrader

const _EXP_VENUE = :SIM
const _EXP_INSTRUMENT = "XYZ"

_exp_bd(seq, ts, side, price, size; recv = ts + 10) =
    BookDelta(_EXP_VENUE, _EXP_INSTRUMENT, ts, recv, seq, side, price, size)
_exp_tp(seq, ts, aggressor, price, size; recv = ts + 10) =
    TradePrint(_EXP_VENUE, _EXP_INSTRUMENT, ts, recv, seq, aggressor, price, size)

# Same 24-event session as experiments/data/tiny_session.jsonl: two-sided book
# throughout, trades on both sides, no crossing or level depletion below zero.
# Events are 50ns apart so an order decided on observation (recv = ts + 10) can
# activate (order_entry 5ns) while later venue events still remain.
function _exp_events()
    return MarketEvent[
        _exp_bd(1, 1000, Bid, 100, 10),
        _exp_bd(2, 1050, Ask, 102, 10),
        _exp_bd(3, 1100, Bid, 101, 5),
        _exp_tp(4, 1150, Buy, 102, 3),
        _exp_bd(5, 1200, Ask, 103, 8),
        _exp_bd(6, 1250, Bid, 100, -3),
        _exp_bd(7, 1300, Ask, 102, 4),
        _exp_tp(8, 1350, Buy, 103, 2),
        _exp_bd(9, 1400, Bid, 101, 6),
        _exp_bd(10, 1450, Ask, 104, 6),
        _exp_tp(11, 1500, Buy, 104, 4),
        _exp_bd(12, 1550, Bid, 100, 8),
        _exp_bd(13, 1600, Ask, 105, 5),
        _exp_tp(14, 1650, Sell, 103, 5),
        _exp_bd(15, 1700, Ask, 104, 3),
        _exp_bd(16, 1750, Bid, 101, -2),
        _exp_tp(17, 1800, Sell, 102, 4),
        _exp_bd(18, 1850, Bid, 101, 9),
        _exp_bd(19, 1900, Ask, 103, 7),
        _exp_bd(20, 1950, Bid, 100, -4),
        _exp_tp(21, 2000, Buy, 103, 6),
        _exp_bd(22, 2050, Ask, 104, 4),
        _exp_bd(23, 2100, Bid, 99, 6),
        _exp_bd(24, 2150, Ask, 105, 5),
    ]
end

function _exp_event_json(event::BookDelta)
    return """{"type":"book_delta","venue":"SIM","instrument":"XYZ","exchange_ts_ns":$(event.exchange_ts_ns),"receive_ts_ns":$(event.receive_ts_ns),"sequence":$(event.sequence),"side":"$(event.side == Bid ? "BID" : "ASK")","price_ticks":$(event.price_ticks),"size_delta":$(event.size_delta)}"""
end
function _exp_event_json(event::TradePrint)
    return """{"type":"trade_print","venue":"SIM","instrument":"XYZ","exchange_ts_ns":$(event.exchange_ts_ns),"receive_ts_ns":$(event.receive_ts_ns),"sequence":$(event.sequence),"aggressor_side":"$(event.aggressor_side == Buy ? "BUY" : "SELL")","price_ticks":$(event.price_ticks),"size":$(event.size)}"""
end

const _EXP_POLICY = """
name = "forecast"
model = "imbalance"
horizon_events = 2
threshold = 0.5
imbalance_threshold = 0.0
order_lots = 2
max_inventory_lots = 4
label_threshold_ticks = 0
"""

const _EXP_SPLITS = """
train_size = 10
validation_size = 4
test_size = 6
embargo = 2
evaluate_on = "test"
"""

const _EXP_OUTCOMES = """
[[outcomes.horizons]]
kind = "events"
n = 3

[[outcomes.horizons]]
kind = "close"

[[outcomes.horizons]]
kind = "session_end"
"""

function _exp_config(
    dir;
    policy = _EXP_POLICY,
    splits = _EXP_SPLITS,
    outcomes = _EXP_OUTCOMES,
    risk = "",
    extra = "",
    header = "schema = \"dendritetrader.experiment.v1\"\nname = \"test_experiment\"\nseed = 7",
)
    session_path = joinpath(dir, "session.jsonl")
    write(session_path, join(_exp_event_json.(_exp_events()), "\n") * "\n")
    config_path = joinpath(dir, "experiment.toml")
    write(
        config_path,
        """
        $(header)

        [data]
        session = "session.jsonl"
        allow_sequence_gaps = false

        [account]
        initial_cash_subunits = 1000000
        mode = "long_flat"
        price_scale = 100
        qty_scale = 1

        [latency]
        order_entry_ns = 5
        cancel_ns = 5
        report_ns = 5

        [fees]
        maker_fee_ppm = 10
        taker_fee_ppm = 10

        [policy]
        $(policy)

        [risk]
        $(risk)

        [splits]
        $(splits)

        [outcomes]
        $(outcomes)
        $(extra)
        """,
    )
    return config_path
end

@testset "chronological_split" begin
    split =
        chronological_split(24; train_size = 10, validation_size = 4, test_size = 6, embargo = 2)
    @test split.train == 1:10
    @test split.validation == 13:16
    @test split.test == 19:24
    @test split.embargo == 2
    @test isempty(intersect(collect(split.train), collect(split.validation)))
    @test isempty(intersect(collect(split.train), collect(split.test)))
    @test isempty(intersect(collect(split.validation), collect(split.test)))
    @test_throws ArgumentError chronological_split(
        24;
        train_size = 0,
        validation_size = 4,
        test_size = 6,
        embargo = 2,
    )
    @test_throws ArgumentError chronological_split(
        24;
        train_size = 12,
        validation_size = 4,
        test_size = 6,
        embargo = 2,
    )
    @test_throws ArgumentError chronological_split(
        24;
        train_size = 10,
        validation_size = 4,
        test_size = 6,
        embargo = -1,
    )
end

@testset "experiment config validation" begin
    mktempdir() do dir
        config = load_experiment_config(_exp_config(dir))
        @test config.name == "test_experiment"
        @test config.seed == UInt64(7)
        @test config.horizon_events == 2
        @test config.embargo == 2
        @test config.evaluate_on == :test
        @test length(config.horizons) == 3
        @test config.horizons[1].kind == :events
        @test config.horizons[1].n == 3
        @test config.horizons[2].kind == :close
        @test config.horizons[3].kind == :session_end

        bad_dir = joinpath(dir, "bad")
        mkpath(bad_dir)

        # Unknown top-level key.
        path = _exp_config(
            bad_dir;
            extra = "",
            header = "schema = \"dendritetrader.experiment.v1\"\nname = \"x\"\nseed = 1\nbogus = 1",
        )
        @test_throws ArgumentError load_experiment_config(path)

        # Missing seed.
        path =
            _exp_config(bad_dir; header = "schema = \"dendritetrader.experiment.v1\"\nname = \"x\"")
        @test_throws ArgumentError load_experiment_config(path)

        # Embargo below the label horizon.
        path = _exp_config(
            bad_dir;
            splits = "train_size = 10\nvalidation_size = 4\ntest_size = 6\nembargo = 1",
        )
        @test_throws ArgumentError load_experiment_config(path)

        # Wrong type.
        path = _exp_config(
            bad_dir;
            splits = "train_size = \"ten\"\nvalidation_size = 4\ntest_size = 6\nembargo = 2",
        )
        @test_throws ArgumentError load_experiment_config(path)

        # Forecast policy without a model.
        path = _exp_config(bad_dir; policy = "name = \"forecast\"\nhorizon_events = 2")
        @test_throws ArgumentError load_experiment_config(path)

        # Unknown horizon kind.
        path = _exp_config(bad_dir; outcomes = "[[outcomes.horizons]]\nkind = \"random_split\"\n")
        @test_throws ArgumentError load_experiment_config(path)

        # Non-TOML file.
        path = joinpath(bad_dir, "experiment.yaml")
        write(path, "schema: x\n")
        @test_throws ArgumentError load_experiment_config(path)
    end
end

@testset "run_experiment deterministic trajectory export" begin
    mktempdir() do dir
        config_path = _exp_config(dir)
        config = load_experiment_config(config_path)
        result = run_experiment(config)
        again = run_experiment(load_experiment_config(config_path))

        @test result.run_id == again.run_id
        @test result.core_hash == again.core_hash
        @test result.dataset_sha256 == bytes2hex(sha256(read(config.session_path)))
        @test result.trajectories == again.trajectories
        @test result.predictions == again.predictions
        @test result.snn == again.snn
        @test result.metrics == again.metrics

        # One row per decision call (one per event).
        @test length(result.calls) == 24
        @test length(result.trajectories) == 24
        @test length(result.snn) == 24

        for row in result.trajectories
            @test row["record_type"] == "paper_trajectory_step"
            @test row["schema_version"] == 1
            @test row["paper_only"] === true
            @test row["run_id"] == result.run_id
            @test haskey(row, "market") &&
                  haskey(row, "signal") &&
                  haskey(row, "decision") &&
                  haskey(row, "paper_account") &&
                  haskey(row, "outcome")
            @test row["phase"] in ("train", "embargo", "validation", "test")
        end

        # With evaluate_on="test", every non-test row is a no-action observation.
        for row in result.trajectories
            row["phase"] == "test" && continue
            @test row["decision"]["action"] == "none"
            @test isempty(row["decision"]["intents"])
        end
        test_rows = [row for row in result.trajectories if row["phase"] == "test"]
        @test length(test_rows) == 6
        @test all(row -> row["info_sequence"] in 19:24, test_rows)

        # The imbalance policy acts on some test rows; every acted row carries a
        # recorded signal and an outcome over each declared horizon.
        acted = [row for row in test_rows if row["decision"]["action"] != "none"]
        @test !isempty(acted)
        for row in acted
            @test row["signal"] !== nothing
            @test row["signal"]["producer"] == "imbalance"
            @test row["signal"]["horizon_events"] == 2
            for name in ("events_3", "close", "session_end")
                @test haskey(row["outcome"]["horizons"], name)
            end
        end
    end
end

@testset "trajectory outcome attaches to the correct decision" begin
    mktempdir() do dir
        result = run_experiment(load_experiment_config(_exp_config(dir)))
        fills = result.simulation.fills
        @test !isempty(fills)
        for row in result.trajectories
            horizons = row["outcome"]["horizons"]
            isempty(horizons) && continue
            session_end = horizons["session_end"]
            decision_fills = [f for f in fills if f.decision_id == row["decision_id"]]
            @test session_end["decision_fills"] == length(decision_fills)
            @test session_end["decision_fees_subunits"] ==
                  sum(f.fee_subunits for f in decision_fills; init = Int64(0))
            @test session_end["decision_net_lots"] == sum(
                f.qty_lots * (f.side == Buy ? 1 : -1) for f in decision_fills;
                init = Int64(0),
            )
        end
        # Rejected decisions carry no fabricated outcome: the risk limit refuses
        # every 2-lot order, so each submit is recorded with accepted=false.
        result_rejected = run_experiment(
            load_experiment_config(
                _exp_config(dir; risk = "max_order_lots = 1", policy = """
                                                              name = "forecast"
                                                              model = "imbalance"
                                                              horizon_events = 2
                                                              threshold = 1.0
                                                              imbalance_threshold = 0.0
                                                              order_lots = 2
                                                              max_inventory_lots = 4
                                                              """),
            ),
        )
        rejected =
            [row for row in result_rejected.trajectories if row["outcome"]["status"] == "rejected"]
        @test !isempty(rejected)
        for row in rejected
            @test isempty(row["outcome"]["horizons"])
            @test !isempty(row["decision"]["intents"])
            @test all(intent -> intent["accepted"] == false, row["decision"]["intents"])
        end
    end
end

@testset "fit sees train rows only" begin
    mktempdir() do dir
        result = run_experiment(load_experiment_config(_exp_config(dir)))
        events = result.session.events
        train_hi = events[last(result.split.train)].sequence
        @test result.model !== nothing
        @test result.model.fitted
        @test result.model.fitted_through_sequence <= train_hi
        @test result.normalizer.fitted
        @test result.normalizer.fitted_through_sequence <= train_hi
        # Split phases: indices 1:10 train, 11:12 embargo, 13:16 validation,
        # 17:18 embargo, 19:24 test.
        @test result.phase[1:10] == fill(:train, 10)
        @test result.phase[11:12] == fill(:embargo, 2)
        @test result.phase[13:16] == fill(:validation, 4)
        @test result.phase[17:18] == fill(:embargo, 2)
        @test result.phase[19:24] == fill(:test, 6)
        @test sort(result.eval_sequences) == Int64[19, 20, 21, 22, 23, 24]
    end
end

@testset "experiment manifest and artifacts" begin
    mktempdir() do dir
        result = run_experiment(load_experiment_config(_exp_config(dir)))
        manifest = result.manifest
        @test manifest["schema"] == "dendritetrader.experiment_manifest.v1"
        @test manifest["paper_only"] === true
        @test manifest["run_id"] == result.run_id
        @test manifest["code"]["julia_version"] == string(VERSION)
        @test manifest["dataset"]["sha256"] == result.dataset_sha256
        @test manifest["dataset"]["n_events"] == 24
        @test manifest["splits"]["train"]["indices"] == [1, 10]
        @test manifest["splits"]["test"]["indices"] == [19, 24]
        @test manifest["splits"]["embargo"] == 2
        @test manifest["seeds"]["experiment"] == "7"
        @test manifest["normalization"]["fitted"] === true
        @test manifest["model"]["name"] == "imbalance"
        @test manifest["hashes"]["simulation_result"] == result.simulation.result_hash
        @test manifest["hashes"]["core"] == result.core_hash
        @test haskey(manifest["assumptions"], "latency")
        @test manifest["diagnostics"]["diagnostic_zero_latency"] === false
        @test manifest["diagnostics"]["diagnostic_zero_cost"] === false
        @test isempty(manifest["reconciliation_errors"])

        outdir = joinpath(dir, "artifacts")
        paths = write_artifacts(result, outdir)
        @test length(paths) == 7
        for name in (
            "manifest.json",
            "metrics.json",
            "predictions.jsonl",
            "trajectory.jsonl",
            "snn_trajectory.jsonl",
            "normalization.json",
            "summary.md",
        )
            @test isfile(joinpath(outdir, name))
        end
        written = JSON.parsefile(joinpath(outdir, "manifest.json"))
        for (name, hash) in written["artifacts"]
            @test bytes2hex(sha256(read(joinpath(outdir, name)))) == hash
        end
        metrics = JSON.parsefile(joinpath(outdir, "metrics.json"))
        @test metrics["run_id"] == result.run_id
        @test metrics["decision_calls"] == 24
        @test metrics["evaluated_calls"] == 6
        @test metrics["phases"]["test"] == 6
        @test metrics["phases"]["embargo"] == 4
        @test metrics["simulation"]["decisions"] == 24
        @test isempty(metrics["reconciliation_errors"])

        trajectory_rows = [
            JSON.parse(line) for
            line in split(strip(read(joinpath(outdir, "trajectory.jsonl"), String)), "\n")
        ]
        @test length(trajectory_rows) == 24
        snn_rows = [
            JSON.parse(line) for
            line in split(strip(read(joinpath(outdir, "snn_trajectory.jsonl"), String)), "\n")
        ]
        @test length(snn_rows) == 24
        for row in snn_rows
            @test row["record_type"] == "snn_trajectory_step"
            @test haskey(row, "features") &&
                  haskey(row, "signal") &&
                  haskey(row, "decision_target") &&
                  haskey(row, "outcome")
            @test row["decision_target"] in ("buy", "sell", "hold", "rejected", "mixed")
        end
        summary = read(joinpath(outdir, "summary.md"), String)
        @test occursin(result.run_id, summary)
        @test occursin("Experiment summary", summary)

        # Artifacts carry no credentials.
        for path in paths
            contents = read(path, String)
            for marker in ("api_key", "apikey", "secret", "token", "password")
                @test !occursin(marker, lowercase(contents))
            end
        end

        # A second run reproduces the deterministic artifacts byte-for-byte
        # (the manifest's generated_utc is the only allowed difference).
        other = joinpath(dir, "artifacts2")
        write_artifacts(result, other)
        for name in (
            "metrics.json",
            "predictions.jsonl",
            "trajectory.jsonl",
            "snn_trajectory.jsonl",
            "normalization.json",
            "summary.md",
        )
            @test read(joinpath(outdir, name)) == read(joinpath(other, name))
        end
    end
end

@testset "zero-latency and external policy labelling" begin
    mktempdir() do dir
        # Zero order-entry latency is legal but must be labelled a diagnostic.
        config_path = _exp_config(dir; extra = "", splits = _EXP_SPLITS)
        toml = read(config_path, String)
        toml = replace(toml, "order_entry_ns = 5" => "order_entry_ns = 0")
        write(config_path, toml)
        result = run_experiment(load_experiment_config(config_path))
        @test result.manifest["diagnostics"]["diagnostic_zero_latency"] === true
        @test "order_entry_ns" in result.manifest["diagnostics"]["zero_latency_components"]

        # External policy: a user-supplied decide callable acts only on eval rows.
        config_path = _exp_config(dir; policy = "name = \"external\"\nhorizon_events = 2")
        calls = Int[]
        result = run_experiment(
            load_experiment_config(config_path);
            decide = ctx -> begin
                push!(calls, ctx.info_sequence)
                first_intent = first(first(ctx.snapshot.asks))
                OrderIntent(Buy, 1, first_intent; time_in_force = :ioc, tag = "t")
            end,
            signal = ctx -> Dict("probe" => ctx.info_sequence),
        )
        @test calls == [19, 20, 21, 22, 23, 24]
        for row in result.trajectories
            if row["phase"] == "test"
                @test row["decision"]["action"] == "submit"
                @test row["signal"]["probe"] == row["info_sequence"]
            else
                @test row["decision"]["action"] == "none"
            end
        end

        # External policy without decide is rejected.
        @test_throws ArgumentError run_experiment(load_experiment_config(config_path))
    end
end
