# SPDX-License-Identifier: MIT OR Apache-2.0

using Test

# Former-owner host fragments are built at runtime so this file does not contain them.
const _FORMER_OWNER = "Limen" * "-Neural"
const _FORMER_PAGES = "limen" * "-neural.github.io"

const _OWNERSHIP_ROOTS = [
    "README.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "REVIEW.md",
    "Project.toml",
    "LICENSE",
    "LICENSE-MIT",
    "LICENSE-APACHE-2.0",
    "src",
    "docs",
    "test",
    ".github",
]

const _SKIP_EXTENSIONS = Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".svg"])

function _ownership_text(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext in _SKIP_EXTENSIONS && return nothing
    bytes = read(path)
    return isvalid(String, bytes) ? String(bytes) : nothing
end

function _ownership_files(root::AbstractString)
    files = String[]
    for relative in _OWNERSHIP_ROOTS
        path = joinpath(root, relative)
        ispath(path) || continue
        if isfile(path)
            push!(files, path)
            continue
        end
        for (dir, _, names) in walkdir(path)
            for name in names
                push!(files, joinpath(dir, name))
            end
        end
    end
    return files
end

@testset "package self-URLs use rmems/DendriteTrader.jl" begin
    root = normpath(joinpath(@__DIR__, ".."))
    hits = String[]
    for path in _ownership_files(root)
        text = _ownership_text(path)
        text === nothing && continue
        if occursin(_FORMER_OWNER, text) || occursin(_FORMER_PAGES, text)
            push!(hits, relpath(path, root))
        end
    end
    @test isempty(hits)

    index = read(joinpath(root, "docs", "src", "index.md"), String)
    @test occursin("https://github.com/rmems/DendriteTrader.jl/actions/workflows/ci.yml", index)
    @test occursin("https://rmems.github.io/DendriteTrader.jl", index)
    @test occursin("] add https://github.com/rmems/DendriteTrader.jl", index)
    @test occursin("https://github.com/rmems/DendriteTrader.jl/blob/main/LICENSE", index)

    readme = read(joinpath(root, "README.md"), String)
    @test occursin("https://github.com/rmems/DendriteTrader.jl", readme)
    @test occursin("https://rmems.github.io/DendriteTrader.jl", readme)

    make = read(joinpath(root, "docs", "make.jl"), String)
    @test occursin("github.com/rmems/DendriteTrader.jl", make)
    @test occursin("https://rmems.github.io/DendriteTrader.jl", make)

    changelog = read(joinpath(root, "CHANGELOG.md"), String)
    @test occursin("https://github.com/rmems/DendriteTrader.jl/releases/tag/v0.1.0", changelog)
end
