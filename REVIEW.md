# Code Review Guidelines

## Thread Safety

For any shared mutable state accessed by multiple tasks or threads (e.g., token buckets, stop flags, caches), protect read-modify-write sequences with `ReentrantLock` or use `Threads.Atomic` types for simple flags.