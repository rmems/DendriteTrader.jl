# Sizing

The `SizingModule` provides Kelly and fractional-Kelly position sizing, mapping neural confidence scores and historical trade statistics to capital allocations.

```julia
using DendriteTrader

fraction = kelly_fraction(win_rate = 0.55, avg_win = 8.50, avg_loss = 5.20)
position = size_position(confidence = 0.90, price = 100.0, account_balance = 10_000.0)
```

## Kelly Helpers

```@docs
DendriteTrader.SizingModule.kelly_fraction
DendriteTrader.SizingModule.half_kelly
DendriteTrader.SizingModule.from_confidence
```

## Risk Classification

```@docs
DendriteTrader.SizingModule.RiskTier
DendriteTrader.SizingModule.risk_tier
```

## Position Sizing

```@docs
DendriteTrader.SizingModule.PositionSize
DendriteTrader.SizingModule.size_position
```

## Position caps in `execute_signal!`

`execute_signal!` sizes entries with `size_position` at its default
`kelly_scalar = 0.5`, so the committed fraction is `0.5 * (p*b - q)/b`, at most `0.5`
of the balance (reached only at confidence `1.0`). The `[0, 0.20]` output cap belongs
to `kelly_fraction`, which `execute_signal!` does not call. For example, confidence
`0.95` with `payoff_ratio = 1.5` commits about `0.458`.

The engine then applies two caps. Neither shrinks an order; a violation rejects it
with a `max_position_reject` event whose reason names the cap.

| Cap | Rule | Default | Reason prefix |
|-----|------|---------|---------------|
| `max_position_fraction` | `units * price <= max_position_fraction * account_balance` | `1.0` (no leverage) | `exceeds_max_position_fraction:` |
| `max_position_size` | `units <= max_position_size` | `Inf` (no absolute cap) | `exceeds_max_position:` |

The default relative cap of `1.0` is a backstop: it is twice the largest Kelly-sized
commitment, so it never rejects a Kelly-sized entry and boundary rounding cannot flip
a decision. A relative cap at or below `0.5` (such as `0.25`) rejects ordinary
high-confidence entries. Orders that only reduce an existing position are exempt from
the relative cap, because price and balance drift after entry. The absolute cap
applies to every order. An explicit `position_units_override` above either cap
throws `ArgumentError` with the same reason.
