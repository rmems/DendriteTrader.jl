# SPDX-License-Identifier: MIT OR Apache-2.0

"""One expanding-window train/validation/test split with explicit embargo gaps."""
struct ChronologicalSplit
    train::UnitRange{Int}
    validation::UnitRange{Int}
    test::UnitRange{Int}
    embargo::Int

    function ChronologicalSplit(
        train::UnitRange{Int},
        validation::UnitRange{Int},
        test::UnitRange{Int},
        embargo::Int,
    )
        embargo >= 0 || throw(ArgumentError("embargo must be non-negative"))
        !isempty(train) && !isempty(validation) && !isempty(test) ||
            throw(ArgumentError("split ranges must be non-empty"))
        first(train) >= 1 || throw(ArgumentError("split indices must be positive"))
        first(validation) >= 1 || throw(ArgumentError("split indices must be positive"))
        first(test) >= 1 || throw(ArgumentError("split indices must be positive"))
        last(train) < first(validation) ||
            throw(ArgumentError("train range must precede validation range"))
        last(validation) < first(test) ||
            throw(ArgumentError("validation range must precede test range"))
        _split_gap(first(validation), last(train)) >= embargo ||
            throw(ArgumentError("train/validation gap is smaller than embargo"))
        _split_gap(first(test), last(validation)) >= embargo ||
            throw(ArgumentError("validation/test gap is smaller than embargo"))
        return new(train, validation, test, embargo)
    end
end

function _split_index(value::Integer, name::AbstractString)
    typemin(Int) <= value <= typemax(Int) || throw(ArgumentError("$(name) must fit in Int"))
    return Int(value)
end

function _split_boundary(left::Int, right::Int)
    try
        return Base.checked_add(left, right)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("split boundary exceeds Int range"))
    end
end

function _split_gap(start_index::Int, end_index::Int)
    try
        return Base.checked_sub(Base.checked_sub(start_index, end_index), 1)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError("split gap exceeds Int range"))
    end
end

"""
    chronological_split(n; train_size, validation_size, test_size, embargo)
        -> ChronologicalSplit

One contiguous split over indices `1:n`: train on the earliest `train_size`
observations, validation on the next `validation_size`, and the final `test_size`
held out last, with an `embargo`-index gap between neighbouring ranges. Indices are
observation positions (for market sequences, event positions), never random
adjacent-row draws. When the declared sizes plus embargo gaps do not cover `n`,
the trailing observations past the test range are unused — they belong to no
split. Returns the single `ChronologicalSplit` contract; the ranges and embargo
are self-validating.
"""
function chronological_split(
    n::Integer;
    train_size::Integer,
    validation_size::Integer,
    test_size::Integer,
    embargo::Integer,
)
    n_int = _split_index(n, "n")
    train_int = _split_index(train_size, "train_size")
    validation_int = _split_index(validation_size, "validation_size")
    test_int = _split_index(test_size, "test_size")
    embargo_int = _split_index(embargo, "embargo")
    train_int > 0 || throw(ArgumentError("train_size must be positive"))
    validation_int > 0 || throw(ArgumentError("validation_size must be positive"))
    test_int > 0 || throw(ArgumentError("test_size must be positive"))
    embargo_int >= 0 || throw(ArgumentError("embargo must be non-negative"))

    train = 1:train_int
    validation_start = _split_boundary(_split_boundary(train_int, embargo_int), 1)
    validation_end = _split_boundary(validation_start, validation_int - 1)
    test_start = _split_boundary(_split_boundary(validation_end, embargo_int), 1)
    test_end = _split_boundary(test_start, test_int - 1)
    test_end <= n_int || throw(ArgumentError("not enough observations for the declared split"))
    return ChronologicalSplit(
        train,
        validation_start:validation_end,
        test_start:test_end,
        embargo_int,
    )
end

"""First row after `last_row` whose event sequence is at least last sequence + embargo + 1."""
function _first_row_after_sequence_embargo(
    row_sequences::AbstractVector{<:Integer},
    last_row::Int,
    embargo::Int,
    n_int::Int,
)
    last_sequence = _split_index(row_sequences[last_row], "row sequence")
    min_sequence = _split_boundary(_split_boundary(last_sequence, embargo), 1)
    start_row = _split_boundary(last_row, 1)
    for row in start_row:n_int
        if _split_index(row_sequences[row], "row sequence") >= min_sequence
            return row
        end
    end
    return nothing
end

"""
    walk_forward_splits(n; train_size, validation_size, test_size, embargo, max_horizon)
    walk_forward_splits(n; train_size, validation_size, test_size, embargo, max_horizon, row_sequences)

Build expanding training windows and disjoint evaluation windows. The embargo must
be at least the maximum feature or label horizon declared by the experiment.

`train_size`, `validation_size`, and `test_size` are always counted in rows.
By default `embargo` and `max_horizon` are also counted in row positions, which is
correct when each row is one event (`FeatureFrame` inputs have strictly increasing
sequences).

Pass `row_sequences` — the event sequence of each row, length `n`, non-decreasing —
to count `embargo` and `max_horizon` in event-sequence distance instead. After a
window ending at row `i` with sequence `s`, the next window starts at the first later
row whose sequence is at least `s + embargo + 1`. Use this when several rows share
one event (per-level or merged trade/book rows) so a horizon declared in events is
not under-covered. `ChronologicalSplit.embargo` still stores the requested embargo
(in the units of that call); the row gap may be larger than that value when
sequences are shared.
"""
function walk_forward_splits(
    n::Integer;
    train_size::Integer,
    validation_size::Integer,
    test_size::Integer,
    embargo::Integer,
    max_horizon::Integer,
    row_sequences::Union{Nothing, AbstractVector{<:Integer}} = nothing,
)
    n_int = _split_index(n, "n")
    train_size_int = _split_index(train_size, "train_size")
    validation_size_int = _split_index(validation_size, "validation_size")
    test_size_int = _split_index(test_size, "test_size")
    embargo_int = _split_index(embargo, "embargo")
    max_horizon_int = _split_index(max_horizon, "max_horizon")
    n_int > 0 || throw(ArgumentError("n must be positive"))
    train_size_int > 0 || throw(ArgumentError("train_size must be positive"))
    validation_size_int > 0 || throw(ArgumentError("validation_size must be positive"))
    test_size_int > 0 || throw(ArgumentError("test_size must be positive"))
    max_horizon_int >= 0 || throw(ArgumentError("max_horizon must be non-negative"))
    embargo_int >= max_horizon_int || throw(ArgumentError("embargo must be at least max_horizon"))

    sequences = nothing
    if row_sequences !== nothing
        length(row_sequences) == n_int || throw(ArgumentError("row_sequences length must equal n"))
        for index in 2:n_int
            row_sequences[index] < row_sequences[index - 1] &&
                throw(ArgumentError("row_sequences must be non-decreasing"))
        end
        sequences = row_sequences
    end

    splits = ChronologicalSplit[]
    train_end = train_size_int
    while true
        if sequences === nothing
            validation_start = _split_boundary(_split_boundary(train_end, embargo_int), 1)
            validation_end = _split_boundary(validation_start, validation_size_int - 1)
            test_start = _split_boundary(_split_boundary(validation_end, embargo_int), 1)
            test_end = _split_boundary(test_start, test_size_int - 1)
        else
            validation_start =
                _first_row_after_sequence_embargo(sequences, train_end, embargo_int, n_int)
            validation_start === nothing && break
            validation_end = _split_boundary(validation_start, validation_size_int - 1)
            validation_end <= n_int || break
            test_start =
                _first_row_after_sequence_embargo(sequences, validation_end, embargo_int, n_int)
            test_start === nothing && break
            test_end = _split_boundary(test_start, test_size_int - 1)
        end
        validation_start <= n_int || break
        validation_end <= n_int || break
        test_start <= n_int || break
        test_end <= n_int || break
        push!(
            splits,
            ChronologicalSplit(
                1:train_end,
                validation_start:validation_end,
                test_start:test_end,
                embargo_int,
            ),
        )
        train_end = test_end
    end
    isempty(splits) && throw(ArgumentError("not enough observations for one complete split"))
    return splits
end
