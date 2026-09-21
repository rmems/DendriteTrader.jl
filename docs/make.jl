# SPDX-License-Identifier: MIT OR Apache-2.0

using Documenter
using DendriteTrader
using DendriteTrader.SizingModule
using DendriteTrader.Backtest

makedocs(
    sitename = "DendriteTrader.jl",
    modules = [DendriteTrader, DendriteTrader.SizingModule, DendriteTrader.Backtest],
    authors = "Raul Montoya Cardenas",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://rmems.github.io/DendriteTrader.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Modules" => "modules.md",
        "Guides" => [
            "Execution" => "execution.md",
            "Sizing" => "sizing.md",
            "Backtest" => "backtest.md",
            "SNN Experiments" => "experiments.md",
        ],
    ],
    checkdocs = :exports,
    warnonly = [:missing_docs],
)

if get(ENV, "DOCUMENTER_TEST", "false") != "true"
    deploydocs(repo = "github.com/rmems/DendriteTrader.jl", devbranch = "main")
end
