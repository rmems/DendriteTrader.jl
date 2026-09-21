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
        _split_gap(first(validation), last(train)) == embargo ||
            throw(ArgumentError("train/validation embargo does not match"))
        _split_gap(first(test), last(validation)) == embargo ||
            throw(ArgumentError("validation/test embargo does not match"))
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
    walk_forward_splits(n; train_size, validation_size, test_size, embargo, max_horizon)

Build expanding training windows and disjoint evaluation windows. The embargo must
be at least the maximum feature or label horizon declared by the experiment.
"""
function walk_forward_splits(
    n::Integer;
    train_size::Integer,
    validation_size::Integer,
    test_size::Integer,
    embargo::Integer,
    max_horizon::Integer,
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

    splits = ChronologicalSplit[]
    train_end = train_size_int
    while true
        validation_start = _split_boundary(_split_boundary(train_end, embargo_int), 1)
        validation_end = _split_boundary(validation_start, validation_size_int - 1)
        test_start = _split_boundary(_split_boundary(validation_end, embargo_int), 1)
        test_end = _split_boundary(test_start, test_size_int - 1)
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
