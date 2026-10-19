# SPDX-License-Identifier: MIT OR Apache-2.0

"""Schema id of the `metabolic-ledger` cross-language accounting fixtures (RM-1863)."""
const PARITY_SCHEMA_ID = "metabolic-ledger.parity.v1"
const _PARITY_SCHEMA_VERSION = 1

_fixture_int(record, field) = begin
    value = get(record, field, nothing)
    value isa Integer && !(value isa Bool) ||
        _fail(:malformed_input, "fixture field $(field) must be an integer")
    Int64(value)
end
_fixture_str(record, field) = begin
    value = get(record, field, nothing)
    value isa AbstractString ||
        _fail(:malformed_input, "fixture field $(field) must be a string")
    String(value)
end

"""Load and schema-check one `metabolic-ledger.parity.v1` JSON fixture."""
function load_parity_fixture(path::AbstractString)
    fixture = JSON.parsefile(path)
    get(fixture, "schema", nothing) == PARITY_SCHEMA_ID ||
        _fail(:malformed_input, "$(path): unknown schema $(get(fixture, "schema", nothing))")
    get(fixture, "schema_version", nothing) == _PARITY_SCHEMA_VERSION ||
        _fail(:malformed_input, "$(path): unsupported schema_version")
    return fixture
end

function _apply_fixture_event!(p::PaperPortfolio, event)
    kind = _fixture_str(event, "kind")
    instrument = _fixture_str(event, "instrument")
    time = _fixture_int(event, "event_time_ns")
    if kind == "fill"
        side_name = _fixture_str(event, "side")
        side =
            side_name == "buy" ? Buy :
            side_name == "sell" ? Sell : _fail(:malformed_input, "unknown side $(side_name)")
        return apply_fill!(
            p,
            _fixture_str(event, "fill_id"),
            time,
            instrument,
            side,
            _fixture_int(event, "price_ticks"),
            _fixture_int(event, "qty_lots"),
            _fixture_int(event, "fee_subunits"),
        )
    elseif kind in ("reject", "cancel", "non_fill")
        record_non_fill!(
            p,
            Symbol(kind),
            _fixture_str(event, "order_id"),
            time,
            instrument,
            _fixture_str(event, "reason"),
        )
        return :applied
    end
    _fail(:malformed_input, "unknown fixture event kind $(kind)")
end

function _compare!(failures, name, expected, actual, tolerance = 0)
    expected === nothing && return failures
    abs(Int128(expected) - Int128(actual)) <= tolerance ||
        push!(failures, "$(name): expected $(expected) ± $(tolerance), got $(actual)")
    return failures
end

"""
    run_parity_fixture(fixture) -> (passed::Bool, failures::Vector{String}, portfolio)

Replay a parity fixture through a long/flat `PaperPortfolio` exactly as the Rust
harness does: apply each event; if `expected.error` is set the last event must fail with
that typed error and leave state unchanged; then compare every expected field with the
declared tolerance (inventory, cost and basis are always exact) and the audit rows.
"""
function run_parity_fixture(fixture)
    failures = String[]
    units = fixture["units"]
    initial = fixture["initial"]
    get(initial, "inventory_lots", 0) == 0 && get(initial, "total_cost_subunits", 0) == 0 ||
        push!(failures, "nonzero initial inventory is outside parity.v1; start flat")
    scale = ParityScale(_fixture_int(units, "price_scale"), _fixture_int(units, "qty_scale"))
    p = PaperPortfolio(
        _fixture_str(units, "instrument"),
        _fixture_int(initial, "cash_subunits");
        scale = scale,
        mode = :long_flat,
    )
    expected = fixture["expected"]
    expected_error = get(expected, "error", nothing)
    events = fixture["events"]
    for (index, event) in enumerate(events)
        before = copy(p)
        try
            _apply_fixture_event!(p, event)
            if index == length(events) && expected_error !== nothing
                push!(failures, "expected error $(expected_error["type"]) but event succeeded")
            end
        catch error
            error isa PortfolioError || rethrow()
            if index == length(events) && expected_error !== nothing
                String(error.kind) == expected_error["type"] || push!(
                    failures,
                    "expected error $(expected_error["type"]) but got $(error.kind)",
                )
                # Reject before mutation: state must equal the pre-event state.
                for field in (
                    :cash_subunits,
                    :inventory_lots,
                    :total_cost_subunits,
                    :realized_pnl_subunits,
                    :fees_subunits,
                )
                    getfield(p, field) == getfield(before, field) ||
                        push!(failures, "failed event mutated $(field)")
                end
                length(p.audit) == length(before.audit) ||
                    push!(failures, "failed event appended an audit row")
            else
                push!(failures, "event $(index - 1): $(sprint(showerror, error))")
                return false, failures, p
            end
        end
    end

    tolerances = get(fixture, "tolerances", Dict{String, Any}())
    tol(name) = Int64(get(tolerances, name, 0))
    mark = get(fixture, "mark_price_ticks", nothing)
    unrealized = mark === nothing ? Int64(0) : unrealized_pnl(p, mark)
    _compare!(
        failures,
        "cash_subunits",
        get(expected, "cash_subunits", nothing),
        p.cash_subunits,
        tol("cash_subunits"),
    )
    _compare!(
        failures,
        "inventory_lots",
        get(expected, "inventory_lots", nothing),
        p.inventory_lots,
    )
    _compare!(
        failures,
        "total_cost_subunits",
        get(expected, "total_cost_subunits", nothing),
        p.total_cost_subunits,
    )
    _compare!(
        failures,
        "weighted_cost_basis_ticks",
        get(expected, "weighted_cost_basis_ticks", nothing),
        weighted_cost_basis_ticks(p),
    )
    _compare!(
        failures,
        "realized_pnl_subunits",
        get(expected, "realized_pnl_subunits", nothing),
        p.realized_pnl_subunits,
        tol("realized_pnl_subunits"),
    )
    _compare!(
        failures,
        "fees_subunits",
        get(expected, "fees_subunits", nothing),
        p.fees_subunits,
        tol("fees_subunits"),
    )
    _compare!(
        failures,
        "unrealized_pnl_subunits",
        get(expected, "unrealized_pnl_subunits", nothing),
        unrealized,
        tol("unrealized_pnl_subunits"),
    )
    expected_audit = get(expected, "audit", nothing)
    if expected_audit !== nothing
        actual_audit = [audit_dict(row) for row in p.audit]
        length(expected_audit) == length(actual_audit) || push!(
            failures,
            "audit length: expected $(length(expected_audit)), got $(length(actual_audit))",
        )
        for (index, (want, got)) in enumerate(zip(expected_audit, actual_audit))
            Set(keys(want)) == Set(keys(got)) ||
                push!(failures, "audit[$(index)]: field set differs")
            for key in keys(want)
                isequal(want[key], get(got, key, missing)) || push!(
                    failures,
                    "audit[$(index)].$(key): expected $(want[key]), got $(get(got, key, missing))",
                )
            end
        end
    end
    return isempty(failures), failures, p
end

"""
    parity_document(result, name; mark_price_ticks=result.metrics.final_mark_price_ticks)

Emit a `metabolic-ledger.parity.v1` document (`source = "dendritetrader.jl"`) from a
long/flat simulation that started flat, so the Rust harness can replay the simulator's
fills and non-fills unchanged. Throws `:unsupported_subset` for signed-mode runs or runs
that did not start flat, instead of clamping anything into the subset.
"""
function parity_document(
    result::SimulationResult,
    name::AbstractString;
    mark_price_ticks::Union{Nothing, Integer} = result.metrics.final_mark_price_ticks,
)
    result.in_parity_subset ||
        _fail(:unsupported_subset, "signed-mode runs are Julia-only and outside parity.v1")
    result.initial_inventory_lots == 0 && result.initial_total_cost_subunits == 0 ||
        _fail(:unsupported_subset, "parity.v1 documents must start flat")
    events = Any[]
    for row in result.audit
        row.idempotent_replay && continue
        if row.kind == :fill
            push!(
                events,
                Dict{String, Any}(
                    "kind" => "fill",
                    "fill_id" => row.fill_id,
                    "event_time_ns" => row.event_time_ns,
                    "instrument" => row.instrument,
                    "side" => _side_string(row.side),
                    "price_ticks" => row.price_ticks,
                    "qty_lots" => row.qty_lots,
                    "fee_subunits" => row.fee_subunits,
                ),
            )
        else
            push!(
                events,
                Dict{String, Any}(
                    "kind" => String(row.kind),
                    "order_id" => row.order_id,
                    "event_time_ns" => row.event_time_ns,
                    "instrument" => row.instrument,
                    "reason" => row.reason,
                ),
            )
        end
    end
    final = last(result.ledger)
    basis =
        final.inventory_lots == 0 ? Int64(0) :
        fld(Int128(final.total_cost_subunits) * result.scale.qty_scale, final.inventory_lots)
    expected = Dict{String, Any}(
        "cash_subunits" => final.cash_subunits,
        "inventory_lots" => final.inventory_lots,
        "total_cost_subunits" => final.total_cost_subunits,
        "weighted_cost_basis_ticks" => Int64(basis),
        "realized_pnl_subunits" => final.realized_pnl_subunits,
        "fees_subunits" => final.fees_subunits,
        "audit" => [audit_dict(row) for row in result.audit],
    )
    if mark_price_ticks !== nothing
        expected["unrealized_pnl_subunits"] =
            final.inventory_lots == 0 ? 0 :
            Int64(
                Int128(mark_price_ticks) * final.inventory_lots ÷ result.scale.qty_scale -
                final.total_cost_subunits,
            )
    end
    document = Dict{String, Any}(
        "schema" => PARITY_SCHEMA_ID,
        "schema_version" => _PARITY_SCHEMA_VERSION,
        "source" => "dendritetrader.jl",
        "name" => String(name),
        "units" => Dict{String, Any}(
            "instrument" => result.instrument,
            "quote" => "QUOTE",
            "price_scale" => result.scale.price_scale,
            "qty_scale" => result.scale.qty_scale,
        ),
        "initial" => Dict{String, Any}(
            "cash_subunits" => result.initial_cash_subunits,
            "inventory_lots" => 0,
            "total_cost_subunits" => 0,
        ),
        "events" => events,
        "mark_price_ticks" => mark_price_ticks,
        "expected" => expected,
        "tolerances" => Dict{String, Any}(
            "cash_subunits" => 0,
            "realized_pnl_subunits" => 0,
            "unrealized_pnl_subunits" => 0,
            "fees_subunits" => 0,
        ),
    )
    return document
end
