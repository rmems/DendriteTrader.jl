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
DendriteTrader.load_history
DendriteTrader.close_log!
```

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
