# SPDX-License-Identifier: MIT OR Apache-2.0

# Replay throughput of the paper simulator against the number of resting orders
# (RM-2236). Paper simulation only; this measures the simulator, not a strategy.
#
#   julia --startup-file=no --project=. benchmark/simulation.jl [events] [samples]
#   julia --startup-file=no --project=. benchmark/simulation.jl 4096 9 prints 1024
#
# With a workload (`prints` or `deltas`) and a resting-order count, only that row
# runs. Prefer that form for numbers you intend to quote: one row per process keeps
# earlier rows' heaps out of the measurement. Timings on a loaded machine are
# meaningless here, so the reported figure is the fastest sample, not the mean.
using DendriteTrader
using JSON
using SHA

# `resting` GTC buy orders join one level, then `count` events arrive. With
# `prints`, every event is a trade print at that level, so each one walks the
# level's queue; otherwise the events are deltas at an unrelated level. Order
# quantities exceed the whole print volume, so no order leaves the queue and the
# resting count stays flat for the whole run.
function workload(resting::Int, count::Int, prints::Bool)
    venue, instrument = :BENCH, "PAPER"
    events = MarketEvent[
        BookDelta(venue, instrument, 10, 10, 1, Bid, 90, 1),
        BookDelta(venue, instrument, 20, 20, 2, Ask, 110, 1),
    ]
    for i in 1:count
        ts = 20 + 10i
        push!(
            events,
            prints ? TradePrint(venue, instrument, ts, ts, i + 2, Sell, 100, 1) :
            BookDelta(venue, instrument, ts, ts, i + 2, Bid, 90, 1),
        )
    end
    session = ReplaySession(events)
    quantity = count + 1
    decide =
        ctx ->
            ctx.info_sequence == 2 ? [OrderIntent(Buy, quantity, 100) for _ in 1:resting] : nothing
    config = SimulationConfig(;
        latency = FixedLatency(1; report_ns = 3),
        limits = RiskLimits(
            max_order_lots = quantity,
            max_inventory_lots = resting * quantity,
            max_initial_cash_subunits = 1_000_000_000_000,
            max_open_orders = resting,
        ),
    )
    run() = simulate!(PaperPortfolio(instrument, 1_000_000_000_000), session, decide; config)
    return run, bytes2hex(sha256(repr(events)))
end

function measure(resting::Int, count::Int, prints::Bool, samples::Int)
    run, data_hash = workload(resting, count, prints)
    run()   # warm up: exclude first-call compilation from the samples
    times, bytes, result = Float64[], Int[], nothing
    for _ in 1:samples
        GC.gc(true)
        sample = @timed run()
        result = sample.value
        push!(times, sample.time)
        push!(bytes, sample.bytes)
    end
    isempty(reconciliation_errors(result)) || error("benchmark run does not reconcile")
    fastest = minimum(times)
    return Dict{String, Any}(
        "workload" => prints ? "same_level_prints" : "unrelated_deltas",
        "resting_orders" => resting,
        "events" => count + 2,
        "fills" => length(result.fills),
        "seconds" => times,
        "fastest_seconds" => fastest,
        "median_seconds" => sort(times)[cld(samples, 2)],
        "events_per_second" => (count + 2) / fastest,
        "allocated_bytes" => minimum(bytes),
        "data_sha256" => data_hash,
        # Identical across revisions unless the fill model or accounting changed.
        "result_hash" => result.result_hash,
    )
end

function main()
    count = length(ARGS) < 1 ? 4096 : parse(Int, ARGS[1])
    samples = length(ARGS) < 2 ? 5 : parse(Int, ARGS[2])
    count > 0 && samples > 0 || error("events and samples must be positive")
    rows = Any[]
    if length(ARGS) >= 4
        ARGS[3] in ("prints", "deltas") || error("workload must be \"prints\" or \"deltas\"")
        push!(rows, measure(parse(Int, ARGS[4]), count, ARGS[3] == "prints", samples))
    else
        for prints in (false, true), resting in (64, 256, 1024)
            push!(rows, measure(resting, count, prints, samples))
        end
    end
    root = dirname(Base.active_project())
    manifest = joinpath(root, "Manifest.toml")
    JSON.print(
        stdout,
        Dict{String, Any}(
            "paper_only" => true,
            "revision" => strip(read(`git -C $root rev-parse HEAD`, String)),
            "dirty" => !isempty(read(`git -C $root status --porcelain`, String)),
            "julia" => string(VERSION),
            "cpu" => Sys.CPU_NAME,
            "machine" => Sys.MACHINE,
            "threads" => Threads.nthreads(),
            "load_average" => Sys.loadavg(),
            "manifest_sha256" => isfile(manifest) ? bytes2hex(open(sha256, manifest)) : nothing,
            "rows" => rows,
        ),
    )
    return println()
end

main()
