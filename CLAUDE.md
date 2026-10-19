@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Purpose, layout, toolchain, the CI command list, and commit conventions are in
`AGENTS.md` (imported above); do not duplicate them here. The review rule is in `REVIEW.md`.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout. The only attempt here was a
`Pkg.instantiate(); Pkg.test()` run on Julia 1.12.7 (Linux, CPU only, throwaway depot); it was stopped by a 500 s
timeout before producing any output (cold dependency download and precompile), so nothing below is verified.
CI runs Julia 1.13 only.

| Command | Status |
|---|---|
| `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` | not run to completion (timed out after 500 s on a cold depot; CI runs this on Julia 1.13) |
| `julia --project=. -e 'using Pkg; Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".", check=true)'` | not run (needs the instantiated environment above; CI `format` job) |
| `julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'` then `julia --project=docs docs/make.jl` | not run (disk nearly full; `docs.yml` runs it) |
| `DYDX_INTEGRATION=true julia --project=. -e 'include("test/integration/test_dydx.jl")'` | not run (calls the public dYdX v4 network; `integration.yml` runs it daily) |

- `Pkg.add("JuliaFormatter")` edits `Project.toml`/`Manifest.toml`; check `git diff` and do not commit that change.
- `Manifest.toml` is gitignored, so `Pkg.test()` resolves fresh each time.
- `Project.toml` lists `JuliaFormatter` under `[extras]` but not in `[targets]`, so it is not a test dependency.

## Architecture map

- `src/DendriteTrader.jl`: package entry point; its module docstring declares the repository boundary.
- `src/market/` (`Events`, `OrderBook`, `Replay`), `src/features/` (`Microstructure`, `Labels`, `Normalization`).
- `src/spikes/` (`DeltaEncoder`, `Interface`), `src/models/` (`Baselines`, `Interface`).
- `src/sizing/SizingModule.jl`, `src/backtest/Backtest.jl`, `src/experiments/` (`Experiments`, `Splits`).
- `test/runtests.jl` plus topic files (`features`, `leakage`, `market_replay`, `models`, `spikes`); fixture in `test/fixtures/book_session.jsonl`.
- `test/integration/test_dydx.jl`: network test, gated by `DYDX_INTEGRATION`.
- `docs/`: Documenter site (`docs/make.jl`, `docs/src/`). `docs/superpowers/plans/` is agent planning history.
- `.agents/setup`, `.devcontainer/`, `.devin/`: agent and container bootstrap, not build steps.

## Invariants and non-goals

- Paper trading and read-only dYdX v4 market data only. Do not add order placement, wallet or key handling, or live execution; docs must not imply profit claims or financial advice.
- Not a production HFT runtime: latency-critical execution belongs to Rust services (`corpus-ipc`), and persistent accounting (`GhostWallet`, PnL) to `metabolic-ledger`.
- Replay and features must stay causal; `test/leakage.jl` guards against look-ahead, so do not weaken it to pass.
- Shared mutable state needs `ReentrantLock` or `Threads.Atomic` (`REVIEW.md`).
- Every `.jl` file (`src/`, `test/`, `docs/make.jl`) starts with `# SPDX-License-Identifier: MIT OR Apache-2.0`. Keep it on new files.
- Format with `.JuliaFormatter.toml` (margin 100); CI checks it.

## Gotchas

- `Project.toml` has no publish `exclude`/`files` list, so adding this file needs no manifest change. Never run `Pkg.register` or any other registry or publish command.
- `.claude/` is not gitignored (`.codex/`, `.cursor/`, `.kilo/`, `.mimo/`, `.mimocode/`, `.beads/` are); do not commit local agent state there.
- CI uses Julia 1.13 while `[compat]` allows 1.10; the integration workflow hits the live public dYdX network.
- Edit only what the task needs.
