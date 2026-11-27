# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Task 6 experiment runner CLI (issue #76 / Linear RM-2118).
#
#   julia --project=. scripts/run_experiment.jl <config.toml> [outdir]
#
# Loads and validates a `dendritetrader.experiment.v1` TOML config, runs the paper
# experiment end to end and writes the artifact set (manifest.json, metrics.json,
# predictions.jsonl, trajectory.jsonl, snn_trajectory.jsonl, normalization.json,
# summary.md). `outdir` defaults to experiments/runs/<run_id>.

using DendriteTrader

function main()
    if length(ARGS) < 1 || length(ARGS) > 2
        println(stderr, "usage: julia --project=. scripts/run_experiment.jl <config.toml> [outdir]")
        return 2
    end
    config = load_experiment_config(ARGS[1])
    result = run_experiment(config)
    outdir = length(ARGS) == 2 ? ARGS[2] : joinpath("experiments", "runs", result.run_id)
    paths = write_artifacts(result, outdir)
    println("run_id: ", result.run_id)
    println("core_hash: ", result.core_hash)
    for path in paths
        println("wrote: ", path)
    end
    errors = result.manifest["reconciliation_errors"]
    isempty(errors) || begin
        println(stderr, "reconciliation errors:")
        foreach(error -> println(stderr, "  ", error), errors)
        return 1
    end
    return 0
end

exit(main())
