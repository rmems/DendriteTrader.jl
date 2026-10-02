# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

`DendriteTrader` is Julia strategy, diagnostics, paper-trading, and control-plane tooling for neural
trading systems (see `README.md`): an experimental, event-driven SNN market-microstructure lab. It
consumes neural trade signals (ZMQ SUB), applies confidence gating, sizes positions with Kelly
helpers, tracks paper positions, and has a read-only dYdX v4 REST client. It is **not** a production
HFT runtime. Latency-critical execution stays in Rust services (e.g. `corpus-ipc`), and persistent
accounting (`GhostWallet`, PnL) belongs to `metabolic-ledger`. The boundary is declared in the module
docstring in `src/DendriteTrader.jl`.

## Layout

| Path | Contents |
|------|----------|
| `src/DendriteTrader.jl` | Package entry point and boundary docstring |
| `src/market/`, `src/features/`, `src/spikes/`, `src/models/` | Market data/replay, features, spike encoders, models |
| `src/sizing/`, `src/backtest/`, `src/experiments/` | Kelly sizing, backtesting, experiment helpers |
| `test/runtests.jl` (+ topic files, `test/fixtures/`) | Unit tests |
| `test/integration/test_dydx.jl` | Live dYdX integration test (network) |
| `docs/` | Documenter.jl site (`docs/make.jl`) |

## Toolchain

- Julia **1.13** in CI (`ci.yml`, `docs.yml`, `integration.yml`); `[compat] julia = "1.10"`.
- No GPU requirement.

## Commands

```bash
# Tests (CI: julia-actions/julia-buildpkg + julia-runtest)
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

# Format check (ci.yml "JuliaFormatter" job). Pkg.add modifies Project.toml/Manifest; don't commit that.
julia --project=. -e 'using Pkg; Pkg.add("JuliaFormatter"); using JuliaFormatter; format(".", check=true)'

# Docs (docs.yml)
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl

# Live integration test (integration.yml; calls the public dYdX v4 network, no secret)
DYDX_INTEGRATION=true julia --project=. -e 'include("test/integration/test_dydx.jl")'
```

## Conventions visible in the repo

- Formatting follows `.JuliaFormatter.toml` (`style = "default"`, `indent = 4`, `margin = 100`) and
  is checked in CI.
- Every source file carries an SPDX license identifier header.
- Code review rules are in `REVIEW.md` (e.g. protect shared mutable state with `ReentrantLock` or
  `Threads.Atomic`). Read it before changing concurrent code.
- `CHANGELOG.md` is maintained. Commit subjects use Conventional Commits or a Linear ID prefix
  (`RM-366: ...`) with the PR number.
- Per the README this is experimental software, not a production HFT runtime, profit claim or
  financial advice. Keep docs consistent with that.
