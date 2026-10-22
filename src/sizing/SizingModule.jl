# SPDX-License-Identifier: MIT OR Apache-2.0

"""
    SizingModule

Self-contained Kelly position sizing submodule for DendriteTrader.

Can be imported independently without pulling in ZMQ, HTTP, or the full execution engine:

```julia
include("src/sizing/SizingModule.jl")
using .SizingModule

fraction = kelly_fraction(win_rate=0.55, avg_win=8.50, avg_loss=5.20)
position = size_position(confidence=0.90, price=100.0, account_balance=10_000.0)
```
"""
module SizingModule

export kelly_fraction, half_kelly, from_confidence
export RiskTier, Aggressive, Moderate, Conservative, Minimal, risk_tier
export PositionSize, size_position

# Input validation helpers. Invalid inputs are rejected with an `ArgumentError`;
# they are never clamped or floored into range (research invariant: unsupported
# sizes/prices are rejected, not coerced).
function _require_probability(name::AbstractString, x::Float64)
    (isfinite(x) && 0.0 <= x <= 1.0) ||
        throw(ArgumentError("$name must be a finite probability in [0, 1], got $(repr(x))"))
    return x
end

function _require_positive(name::AbstractString, x::Float64)
    (isfinite(x) && x > 0.0) ||
        throw(ArgumentError("$name must be finite and positive, got $(repr(x))"))
    return x
end

function _require_nonnegative(name::AbstractString, x::Float64)
    (isfinite(x) && x >= 0.0) ||
        throw(ArgumentError("$name must be finite and non-negative, got $(repr(x))"))
    return x
end

"""
    kelly_fraction(; win_rate, avg_win, avg_loss) -> Float64

Compute a conservative half-Kelly position fraction from historical trade statistics.

# Arguments
- `win_rate`: fraction of trades that were profitable; must be finite and in `[0.0, 1.0]`
- `avg_win`: average profit of winning trades; must be finite and `> 0`
- `avg_loss`: average loss of losing trades as an absolute value; must be finite and `> 0`

The derived ratio `avg_win / avg_loss` must also be finite and `> 0` (it can overflow
or underflow even when both inputs are valid).

# Returns
Half-Kelly fraction capped to `[0.0, 0.20]`. The output cap is documented sizing
policy: a negative edge returns `0.0` and the fraction never exceeds `0.20`.

# Throws
`ArgumentError` for any invalid input. Behavior change: earlier versions silently
clamped `win_rate` into `[0, 1]` and floored `avg_win`/`avg_loss` at `1e-9`.
"""
function kelly_fraction(; win_rate::Real, avg_win::Real, avg_loss::Real)::Float64
    win_rate = _require_probability("win_rate", Float64(win_rate))
    avg_win = _require_positive("avg_win", Float64(avg_win))
    avg_loss = _require_positive("avg_loss", Float64(avg_loss))

    # The derived payoff ratio can overflow to Inf or underflow to 0 even when both
    # inputs are finite and positive (e.g. 1e308 / 1e-308); that would make the Kelly
    # formula NaN, so reject it rather than return an uncapped or undefined fraction.
    b = avg_win / avg_loss
    (isfinite(b) && b > 0.0) || throw(
        ArgumentError(
            "avg_win / avg_loss must be a finite positive ratio, got $(repr(b)) " *
            "from avg_win=$(repr(avg_win)), avg_loss=$(repr(avg_loss))",
        ),
    )
    q = 1.0 - win_rate
    full = (win_rate * b - q) / b
    half = full * 0.5

    return clamp(half, 0.0, 0.20)
end

"""
    half_kelly(p, b) -> Float64

Compute the half-Kelly fraction for win probability `p` and payoff ratio `b`.

`p` must be finite and in `[0, 1]`; `b` must be finite and `> 0`. The result is
capped to `[0.0, 1.0]` (documented sizing policy: a negative edge returns `0.0`).

# Throws
`ArgumentError` for any invalid input. Behavior change: earlier versions silently
clamped `p` into `[0, 1]` and floored `b` at `1e-9`.
"""
function half_kelly(p::Real, b::Real)::Float64
    p = _require_probability("p", Float64(p))
    b = _require_positive("b", Float64(b))
    q = 1.0 - p
    full = (p * b - q) / b
    return clamp(full * 0.5, 0.0, 1.0)
end

"""
    from_confidence(; confidence, payoff_ratio=1.5) -> Float64

Map neural confidence `[0, 1]` to a half-Kelly position fraction.

`payoff_ratio` is interpreted as an odds-style average win/loss ratio. Delegates to
[`half_kelly`](@ref), so a confidence outside `[0, 1]` (or non-finite) or a
non-positive payoff ratio throws `ArgumentError` instead of being clamped.
"""
function from_confidence(; confidence::Real, payoff_ratio::Real = 1.5)::Float64
    return half_kelly(confidence, payoff_ratio)
end

"""
    @enum RiskTier

Qualitative risk classification derived from neural confidence.
"""
@enum RiskTier begin
    Aggressive = 4
    Moderate = 3
    Conservative = 2
    Minimal = 1
end

"""
    risk_tier(confidence) -> RiskTier

Classify neural confidence into a qualitative risk tier.
"""
function risk_tier(confidence::Real)::RiskTier
    confidence = Float64(confidence)
    if confidence >= 0.95
        return Aggressive
    elseif confidence >= 0.85
        return Moderate
    elseif confidence >= 0.70
        return Conservative
    else
        return Minimal
    end
end

"""
    PositionSize

Result of a Kelly position sizing calculation.

# Fields
- `units`: number of asset units to trade
- `kelly_fraction`: fraction of account balance committed
- `confidence`: original neural confidence input
- `risk`: qualitative risk tier
- `account_risk_pct`: percentage of account balance committed
"""
struct PositionSize
    units::Float64
    kelly_fraction::Float64
    confidence::Float64
    risk::RiskTier
    account_risk_pct::Float64
end

"""
    size_position(; confidence, price, account_balance, payoff_ratio=1.5, kelly_scalar=0.5) -> PositionSize

Compute position sizing from neural confidence.

`payoff_ratio` is interpreted as an odds-style average win/loss ratio.

# Input requirements
- `confidence`: finite, in `[0, 1]`
- `price`: finite, `> 0`
- `account_balance`: finite, `> 0`
- `payoff_ratio`: finite, `> 0`
- `kelly_scalar`: finite, `>= 0`. `kelly_scalar = 0` is valid and yields a zero-unit
  position (`execute_signal!` reports it as `zero_reject`).

The resulting Kelly fraction is capped to `[0, 1]` (documented sizing policy: a
negative edge yields a zero-unit position, which is a real sizing decision).

# Throws
`ArgumentError` for any invalid input. Behavior change: earlier versions returned a
zero-unit `PositionSize` with no reason when `price <= 0` or `account_balance <= 0`,
clamped `confidence` into `[0, 1]`, and floored `payoff_ratio` at `1e-9`. Also
throws if the derived unit count `k * account_balance / price` overflows.
"""
function size_position(;
    confidence::Real,
    price::Real,
    account_balance::Real,
    payoff_ratio::Real = 1.5,
    kelly_scalar::Real = 0.5,
)::PositionSize
    p = _require_probability("confidence", Float64(confidence))
    price = _require_positive("price", Float64(price))
    account_balance = _require_positive("account_balance", Float64(account_balance))
    b = _require_positive("payoff_ratio", Float64(payoff_ratio))
    kelly_scalar = _require_nonnegative("kelly_scalar", Float64(kelly_scalar))

    q = 1.0 - p
    full_k = (p * b - q) / b
    k = clamp(full_k * kelly_scalar, 0.0, 1.0)

    units = (k * account_balance) / price
    # The derived unit count can overflow even when every input is finite and valid.
    isfinite(units) || throw(
        ArgumentError(
            "derived units must be finite, got $(repr(units)) from " *
            "account_balance=$(repr(account_balance)), price=$(repr(price))",
        ),
    )
    risk_pct = (units * price) / account_balance * 100.0

    return PositionSize(units, k, p, risk_tier(p), risk_pct)
end

end # module
