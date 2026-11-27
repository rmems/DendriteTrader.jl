# SPDX-License-Identifier: MIT OR Apache-2.0

# Artifact writer for `run_experiment` output (Task 6 / issue #76). Every artifact
# is canonical JSON (or canonical-JSON lines), written atomically via a temp file
# then rename. The manifest is written last and records the SHA-256 of every other
# artifact, so a result set verifies end to end.

const ARTIFACT_FILES = (
    "manifest.json",
    "metrics.json",
    "predictions.jsonl",
    "trajectory.jsonl",
    "snn_trajectory.jsonl",
    "normalization.json",
    "summary.md",
)

function _write_atomic(path::AbstractString, contents::AbstractString)
    temporary = path * ".tmp"
    write(temporary, contents)
    mv(temporary, path; force = true)
    return nothing
end

_jsonl(rows) = isempty(rows) ? "" : join((_canon(row) for row in rows), "\n") * "\n"

function _normalization_doc(result::ExperimentResult)
    normalizer = result.normalizer
    normalizer.fitted || return Dict{String, Any}(
        "schema" => "dendritetrader.experiment_normalization.v1",
        "paper_only" => true,
        "run_id" => result.run_id,
        "fitted" => false,
        "reason" => "no complete-book feature rows inside the train split",
    )
    params = normalization_parameters(normalizer)
    return Dict{String, Any}(
        "schema" => "dendritetrader.experiment_normalization.v1",
        "paper_only" => true,
        "run_id" => result.run_id,
        "fitted" => true,
        "fit_scope" => "train split feature rows only; never validation or test",
        "parameters" => params,
    )
end

function _summary_md(result::ExperimentResult)
    config = result.config
    metrics = result.simulation.metrics
    diagnostics = result.manifest["diagnostics"]
    lines = String[
        "# Experiment summary: $(config.name)",
        "",
        "- Run: `$(result.run_id)`",
        "- Core hash: `$(result.core_hash)`",
        "- Dataset: `$(config.session_path)` (sha256 `$(result.dataset_sha256)`)",
        "- Events: $(length(result.session)); decisions: $(length(result.calls))",
        "- Split: train $(first(result.split.train)):$(last(result.split.train)), " * "validation $(first(result.split.validation)):$(last(result.split.validation)), " * "test $(first(result.split.test)):$(last(result.split.test)) " * "(event positions, embargo $(result.split.embargo)); evaluate_on `$(config.evaluate_on)`",
    ]
    if result.model !== nothing
        push!(
            lines,
            "- Model: `$(config.model)` fitted through sequence " *
            "$(result.model.fitted_through_sequence) (train split only)",
        )
    end
    push!(
        lines,
        "- Normalization: " * (
            result.normalizer.fitted ? "RollingZScore fitted on train feature rows" :
            "not fitted (no train feature rows)"
        ),
    )
    append!(
        lines,
        [
            "",
            "## Diagnostics",
            "",
            "- Zero-latency components: " *
            (
                isempty(diagnostics["zero_latency_components"]) ? "none" :
                join(diagnostics["zero_latency_components"], ", ")
            ) *
            (diagnostics["diagnostic_zero_latency"] ? " — diagnostic run, not evidence" : ""),
            "- Zero-cost run (no fees or slippage): $(diagnostics["diagnostic_zero_cost"])" * (
                diagnostics["diagnostic_zero_cost"] ?
                " — diagnostic run, not a profitability claim" : ""
            ),
            "- Reconciliation errors: $(length(result.manifest["reconciliation_errors"]))",
        ],
    )
    append!(
        lines,
        [
            "",
            "## Simulation metrics",
            "",
            "- Orders submitted: $(metrics.orders_submitted) " *
            "(accepted $(metrics.orders_accepted), rejected $(metrics.orders_rejected), " *
            "cancelled $(metrics.orders_cancelled), expired $(metrics.orders_expired))",
            "- Fills: $(metrics.fills) ($(metrics.maker_fills) maker / $(metrics.taker_fills) taker), " *
            "fill rate $(round(metrics.fill_rate; digits = 4))",
            "- Turnover: $(metrics.turnover_subunits) subunits; fees: $(metrics.fees_subunits); " *
            "slippage: $(metrics.slippage_subunits)",
            "- Realized PnL: $(metrics.realized_pnl_subunits) subunits; " *
            "final equity: $(metrics.final_equity_subunits === nothing ? "unknown (no mark)" : metrics.final_equity_subunits); " *
            "max drawdown: $(metrics.max_drawdown_subunits)",
        ],
    )
    outcomes = result.metrics["outcomes"]
    if !isempty(outcomes)
        push!(
            lines,
            "",
            "## Outcome aggregates",
            "",
            "| horizon | count | total pnl (subunits) | mean pnl | win rate |",
            "|---|---|---|---|---|",
        )
        for name in sort!(collect(keys(outcomes)))
            aggregate = outcomes[name]
            push!(
                lines,
                "| $name | $(aggregate["count"]) | $(aggregate["total_pnl_subunits"]) | " *
                "$(round(aggregate["mean_pnl_subunits"]; digits = 2)) | " *
                "$(round(aggregate["win_rate"]; digits = 4)) |",
            )
        end
    end
    append!(
        lines,
        [
            "",
            "---",
            "",
            "Experimental research output on replayed data. Fills are L2 approximations. " *
            "Not a production HFT runtime, profit claim or financial advice.",
        ],
    )
    return join(lines, "\n") * "\n"
end

"""
    write_artifacts(result::ExperimentResult, dir) -> Vector{String}

Write the Task 6 artifact set under `dir` (created if needed): `manifest.json`,
`metrics.json`, `predictions.jsonl`, `trajectory.jsonl`, `snn_trajectory.jsonl`,
`normalization.json` and `summary.md`. Every write is atomic (temp file then
rename); the manifest records the SHA-256 of each other artifact under
`"artifacts"` before it is itself written. Returns the written paths.
"""
function write_artifacts(result::ExperimentResult, dir::AbstractString)
    mkpath(dir)
    written = Dict{String, String}()
    payloads = Dict{String, String}(
        "metrics.json" => _canon(result.metrics) * "\n",
        "predictions.jsonl" => _jsonl(result.predictions),
        "trajectory.jsonl" => _jsonl(result.trajectories),
        "snn_trajectory.jsonl" => _jsonl(result.snn),
        "normalization.json" => _canon(_normalization_doc(result)) * "\n",
        "summary.md" => _summary_md(result),
    )
    for (name, payload) in payloads
        path = joinpath(dir, name)
        _write_atomic(path, payload)
        written[name] = path
    end
    artifact_hashes =
        Dict{String, Any}(name => bytes2hex(sha256(payload)) for (name, payload) in payloads)
    result.manifest["artifacts"] = artifact_hashes
    manifest_path = joinpath(dir, "manifest.json")
    _write_atomic(manifest_path, _canon(result.manifest) * "\n")
    written["manifest.json"] = manifest_path
    return [written[name] for name in ARTIFACT_FILES]
end
