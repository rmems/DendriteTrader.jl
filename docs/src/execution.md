# Execution

The core `DendriteTrader` module provides the execution engine, ZMQ signal listener, dYdX v4 REST client, price cache, and rate limiter.

## Signal Types

```@docs
DendriteTrader.TradeSide
DendriteTrader.TradeSignal
DendriteTrader.SignalEvent
DendriteTrader.ExecutionDecision
```

## Signal Validation and Processing

```@docs
DendriteTrader.validate_signal
DendriteTrader.latency_ns
DendriteTrader.passes_gate
```

## Execution Engine

```@docs
DendriteTrader.ExecutionEngine
DendriteTrader.execute_signal!
DendriteTrader.start!
DendriteTrader.stop!
DendriteTrader.events
DendriteTrader.fill_rate
DendriteTrader.load_config
DendriteTrader.load_history
DendriteTrader.close_log!
```

## File configuration

`load_config(path)` creates an `ExecutionEngine` from a flat TOML or YAML file.
Only the constructor keys below are accepted; omitted keys retain their defaults.

| Key | Type | Default |
|-----|------|---------|
| `confidence_threshold` | number | `0.85` |
| `max_position_size` | number | `10.0` |
| `payoff_ratio` | number | `1.5` |
| `log_file` | string or YAML `null` | no event log |
| `truncate` | boolean | `false` |

```toml
# engine.toml
confidence_threshold = 0.9
max_position_size = 25.0
payoff_ratio = 1.75
log_file = "events.jsonl"
truncate = true
```

```yaml
# engine.yaml
confidence_threshold: 0.9
max_position_size: 25.0
payoff_ratio: 1.75
log_file: events.jsonl
truncate: true
```

```julia
engine = load_config("engine.toml")
```

Unknown keys, non-mapping documents, and values with the wrong type throw an
`ArgumentError`; malformed TOML or YAML propagates its parser error.

## dYdX v4 REST Client

```@docs
DendriteTrader.DydxClient
DendriteTrader.DydxPrice
DendriteTrader.get_price
DendriteTrader.mid_price
DendriteTrader.spread_bps
```

## Rate Limiter

```@docs
DendriteTrader.RateLimiter
DendriteTrader.acquire!
DendriteTrader.set_rate!
```

## Price Cache

```@docs
DendriteTrader.PriceCache
DendriteTrader.get_cached
DendriteTrader.put_cached!
DendriteTrader.invalidate!
DendriteTrader.clear!
DendriteTrader.cache_size
DendriteTrader.is_fresh
```
