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
        first(validation) - last(train) - 1 == embargo ||
            throw(ArgumentError("train/validation embargo does not match"))
        first(test) - last(validation) - 1 == embargo ||
            throw(ArgumentError("validation/test embargo does not match"))
        return new(train, validation, test, embargo)
    end
end

function _split_index(value::Integer, name::AbstractString)
    typemin(Int) <= value <= typemax(Int) || throw(ArgumentError("$name must fit in Int"))
    return Int(value)
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
    n > 0 || throw(ArgumentError("n must be positive"))
    train_size > 0 || throw(ArgumentError("train_size must be positive"))
    validation_size > 0 || throw(ArgumentError("validation_size must be positive"))
    test_size > 0 || throw(ArgumentError("test_size must be positive"))
    max_horizon >= 0 || throw(ArgumentError("max_horizon must be non-negative"))
    embargo >= max_horizon || throw(ArgumentError("embargo must be at least max_horizon"))

    n = _split_index(n, "n")
    train_size = _split_index(train_size, "train_size")
    validation_size = _split_index(validation_size, "validation_size")
    test_size = _split_index(test_size, "test_size")
    embargo = _split_index(embargo, "embargo")

    splits = ChronologicalSplit[]
    train_end = Int(train_size)
    while true
        validation_start = train_end + embargo + 1
        validation_end = validation_start + validation_size - 1
        test_start = validation_end + embargo + 1
        test_end = test_start + test_size - 1
        test_end <= n || break
        push!(
            splits,
            ChronologicalSplit(
                1:train_end,
                validation_start:validation_end,
                test_start:test_end,
                Int(embargo),
            ),
        )
        train_end = test_end
    end
    isempty(splits) && throw(ArgumentError("not enough observations for one complete split"))
    return splits
end
