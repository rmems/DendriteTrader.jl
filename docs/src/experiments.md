# SNN Market-Microstructure Experiments

DendriteTrader is an experimental market-microstructure and paper-execution
laboratory. It evaluates whether spike-based models produce useful, calibrated
forecasts from causally available market events after declared latency, fill, fee,
and risk assumptions.

It is not a production HFT runtime, an authenticated exchange client, or a source of
financial advice. Venue-critical execution remains outside this Julia package.

## Experiment boundary

The research pipeline is intentionally decomposed:

```text
market events -> causal features -> spike encoder -> forecast model
              -> calibration -> policy -> paper simulator -> evidence artifacts
```

DendriteTrader owns finance-specific schemas, features, labels, policies, paper
execution, and experiment orchestration. Reusable neuron dynamics, plasticity, and
generic SNN training belong in dedicated neuromorphic packages and connect through
adapters. Deterministic low-latency IPC and real venue execution belong in separate
services.

## Scientific requirements

Every reported experiment must preserve:

- a chronological train/validation/test split with an embargo at least as long as
  the maximum feature or forecast horizon;
- training-only normalization and calibration;
- explicit event-time and physical-time horizons;
- non-SNN baselines evaluated on the same decisions and market events;
- declared fees, latency, fill, and queue assumptions;
- per-session metrics and aggregate uncertainty, not only one headline total;
- configuration, data hashes, code revision, seeds, environment, and model identity;
- negative and null findings;
- no credentials, wallet material, or live order submission.

## Evidence families

Experiments report four complementary groups of measurements:

1. Predictive: MCC, macro F1, balanced accuracy, and per-class precision/recall.
2. Calibration: Brier score, expected calibration error, and reliability bins.
3. Paper execution: net PnL, implementation shortfall, turnover, fill rate,
   inventory exposure, drawdown, and completed-transaction correctness.
4. Systems/SNN: encoding and inference latency percentiles, throughput, spike
   density, silent or saturated channels, memory, and a clearly labeled energy proxy
   or hardware measurement.

Classification performance alone is not evidence of an actionable strategy.

## Initial campaign

The first campaign compares a queue-imbalance rule, a linear classifier, a
uniform-time-constant LIF network, and a mixed-memory LIF network. All models consume
the same causal book-derived features and are evaluated at the same decision times.

The campaign starts with deterministic synthetic fixtures in CI. Real-data runs use
separately acquired data with a session manifest recording license, venue,
instrument, timestamp semantics, depth, parser revision, raw hashes, sequence-gap
policy, and known limitations. Raw licensed or large market data is not committed.

Zero-cost and zero-latency scenarios are diagnostics, not realistic claims. L2 data
may support an explicitly approximate queue model; queue-accurate claims require
suitable order-level data.

## Canonical replay input

The replay kernel accepts newline-delimited JSON containing `book_delta` and
`trade_print` events. Prices and sizes use integer ticks and units; floating-point
prices are deliberately excluded from this boundary. Every record identifies one
venue and instrument and carries positive exchange/receive nanosecond timestamps
plus a positive sequence number.

```json
{"type":"book_delta","venue":"SIM","instrument":"XYZ","exchange_ts_ns":1000,"receive_ts_ns":1010,"sequence":1,"side":"BID","price_ticks":100,"size_delta":10}
{"type":"trade_print","venue":"SIM","instrument":"XYZ","exchange_ts_ns":1001,"receive_ts_ns":1011,"sequence":2,"aggressor_side":"BUY","price_ticks":102,"size":3}
```

```julia
session = load_session_jsonl("session.jsonl")
book = OrderBookState(session.venue, session.instrument)
snapshots = replay!(book, session)
```

Replay is fail-closed by default: sequences must strictly increase without gaps,
exchange and receive timestamps cannot move backward, depth cannot become negative,
and an update cannot cross the book. Batch replay is transactional: any invalid
event leaves the destination book unchanged. A feed known to omit events or use
non-contiguous sequence values may use `ReplayPolicy(allow_sequence_gaps=true)`
explicitly; that assumption remains part of the `ReplaySession`. Trade prints
advance event time and sequence but do not invent unobserved changes to L2 depth.

## Causal dataset layer

`microstructure_features` emits one row only after both sides of the book exist. Each
row contains the integer spread, mid-price, size-weighted microprice, top-level
imbalance, and signed top-of-book order flow. The calculation reads the current and
immediately preceding complete snapshots only; it rejects mixed venues, mixed
instruments, and non-monotonic event streams. Sequence gaps are rejected by
default because signed order flow requires adjacent exchange events. Set
`allow_sequence_gaps=true` only when the input is intentionally an observed-event
stream.

`RollingZScore` is fit explicitly on a supplied training `FeatureFrame` and records
the final training sequence. `transform!` never updates those fitted parameters, so
validation and test observations cannot change the training normalization evidence.
`normalization_parameters` returns the feature order, means, scales, and fitting
boundary in JSON-serializable form.

`label_event_horizon` compares an anchor with the snapshot at the exact future event
offset in the original stream. It omits incomplete anchors or exact targets rather
than compressing time, and each `MovementTarget` records both sequences for
unambiguous alignment. Sequence gaps are rejected by default. Labels never cross the
supplied session boundary. `walk_forward_splits` constructs expanding training
windows with disjoint validation/test windows and explicit embargo gaps; it rejects
an embargo shorter than the declared maximum feature or label horizon.

## Tiny CPU baseline

This no-network, no-GPU example fits a stationary baseline on chronological training
data, then returns a normalized three-class forecast:

```julia
using DendriteTrader

rows = FeatureFrame([FeatureRow(10, 1, 1, 100, 100, -0.2, 0),
                     FeatureRow(20, 2, 1, 101, 101, 0.3, 1)])
model = fit!(StationaryModel(), rows, [MovementTarget(1, 2, 1, Up)])
forecast = only(predict!(model, FeatureFrame([rows[1]]), ForecastHorizon(1)))
```

`Forecast.probabilities` use `(Down, Flat, Up)` order. They are class probabilities,
not calibrated win probabilities, and must not be passed directly to Kelly sizing.

## Latency-aware paper simulation

`simulate!(portfolio, session, decide; config)` replays one `ReplaySession` as a
deterministic event-driven **paper** simulation. It never submits orders, reads
credentials, or touches a network, and its fills are hypothetical inferences from L2
data, not observed executions.

Four clocks are kept separate for every order: the time the information used by the
decision became observable (`info_available_ts_ns`, from the feed-delay model —
`ReceiveTimestampFeed` or `FixedFeedDelay`), the decision time, the activation time
(`decision + FixedLatency.order_entry_ns`), and each fill time (never earlier than
activation). Cancels take effect after `cancel_ns`, and fills become visible to the
decision function only after `report_ns`.

Zero latency is allowed only as a labelled diagnostic. If any of `order_entry_ns`,
`cancel_ns` or `report_ns` is zero, `assumptions["latency"]["diagnostic_zero_latency"]` is
`true` and `assumptions["latency"]["zero_latency_components"]` names the zero components;
`is_zero_latency_diagnostic(result)` reads the flag. All three trigger it because each
removes a delay that favours the strategy: zero order entry fills at the decision instant,
zero cancel withdraws orders before any venue event can reach them, and zero report shows
the decision its own fills immediately. The flag is part of `result_hash`, and such a run
must not be reported as evidence. Feed delay is recorded separately under
`assumptions["feed"]` and does not set this flag.

`decide(ctx::DecisionContext)` sees only: the decision time; the newest observable
event (`event`, its `info_sequence` and `info_exchange_ts_ns`); a private copy of the
observed book snapshot; the causal `microstructure_features` row for it; reported
inventory and cash; and reported working orders. Mutating `ctx.snapshot` cannot change
later features or fills.

The hard guarantee against label leakage is structural: `DecisionContext` has no label,
target, or outcome fields, and reading any other property throws. In addition, a
best-effort runtime scan refuses a decision closure or callable struct that captures
`MovementTarget` values within 4 nested levels (the closure's captured variable is
level 1). The scan does not see globals or deeper
nesting; it is a guard against accidental capture, not a sandbox.
`ForecastPolicy` is a minimal example that wires a fitted Task 4 forecast model into
long/flat IOC orders; it is not a strategy claim.

`L2QueueApproximation` is an explicitly approximate fill model (compare the
hftbacktest order-fill documentation). Its rules are copied into
`SimulationResult.assumptions`: marketable orders sweep displayed liquidity within
the limit; consumed displayed liquidity is not reused; resting orders join the back
of their level (or a seeded position with `queue_position = :random`, never ahead of an
earlier simulated order at that level); prints at the order's price advance the queue,
and a displayed-size decrease advances it only when the level becomes smaller than the
venue size still ahead; simulated orders at one level are first-in, first-out by join
order; a print walks the hit side from the best price to the print price, filling
better-priced queues first; quote-throughs fill better-priced resting orders first;
orders whose activation falls after the last replayed event expire unfilled; and there
is no market impact or matching-engine guarantee.

`FeeSchedule` applies one integer fee per fill (positive commission, negative rebate,
parts-per-million of notional, commissions rounded up). Declared taker slippage is in
the fill price and reported separately; neither is charged again.

`PaperPortfolio` is an integer evaluation state, not an authoritative ledger. In the
default `:long_flat` mode it follows the `metabolic-ledger` accounting contract
exactly (single instrument, inventory never negative, sells beyond inventory rejected
as `unsupported_short`, truncating cost allocation, full close takes remaining cost).
`:signed` mode is a Julia-only research extension; its results are labelled outside
the parity subset and `parity_document` refuses to export them. Marks use only the
venue mid at or before each step (rounded half a tick against the position); a
one-sided book yields a `:stale` mark and no prior mid yields `:missing` (no equity is
invented). `reconciliation_errors` independently re-derives cash, inventory, fees,
unrealized PnL, equity and drawdown at every step.

The vendored `metabolic-ledger.parity.v1` fixtures (with `PROVENANCE.json` recording
the source commit and SHA-256 of each file) are replayed by `run_parity_fixture`, and
`parity_document` emits a long/flat simulation in the same schema for the Rust
harness. `decision_records` and `fill_records` emit normalized JSON-ready records for
a later trajectory export; that export is not implemented here.

## API reference

```@docs
BookDelta
TradePrint
BookSnapshot
ReplayPolicy
ReplaySession
OrderBookState
apply!
snapshot
best_bid
best_ask
spread_ticks
replay!
load_session_jsonl
FeatureRow
FeatureFrame
microstructure_features
RollingZScore
fit!
transform!
normalization_parameters
label_event_horizon
MovementTarget
ChronologicalSplit
walk_forward_splits
ForecastHorizon
Forecast
StationaryModel
ImbalanceRule
RidgeClassifier
DeltaEncoder
SpikeFrame
encode!
spike_density
reset_state!
OrderIntent
CancelIntent
RiskLimits
PaperOrder
OpenOrderView
FixedLatency
ReceiveTimestampFeed
FixedFeedDelay
FeeSchedule
L2QueueApproximation
Fill
ParityScale
PortfolioError
PaperPortfolio
apply_fill!
record_non_fill!
SimulationConfig
DecisionContext
DecisionRecord
LedgerPoint
SimulationMetrics
SimulationResult
simulate!
ForecastPolicy
reconciliation_errors
zero_latency_components
is_zero_latency_diagnostic
decision_records
fill_records
run_parity_fixture
parity_document
load_parity_fixture
PARITY_SCHEMA_ID
AuditRow
audit_dict
AbstractFeedDelay
activation_time
cancel_effective_time
available_time
fill_fee
notional_subunits
unrealized_pnl
weighted_cost_basis_ticks
in_parity_subset
reconciles
```
