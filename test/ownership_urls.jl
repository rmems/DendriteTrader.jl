# SPDX-License-Identifier: MIT OR Apache-2.0

using Test

# Former-owner URL fragments are built at runtime so this file does not contain them.
const _FORMER_OWNER = lowercase("Limen" * "-Neural")
const _FORMER_GITHUB_URL =
    Regex(raw"(?:^|[^A-Za-z0-9.-])github\.com[:/]" * _FORMER_OWNER * raw"(?:[/:?#]|$)", "i")
const _FORMER_PAGES_URL =
    Regex(raw"(?:^|[^A-Za-z0-9.-])" * _FORMER_OWNER * raw"\.github\.io(?:[/:?#]|$)", "i")

const _SKIP_EXTENSIONS = Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".svg"])

function _ownership_text(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext in _SKIP_EXTENSIONS && return nothing
    bytes = read(path)
    return isvalid(String, bytes) ? String(bytes) : nothing
end

function _ownership_excluded(path::AbstractString, root::AbstractString)
    relative = replace(relpath(path, root), "\\" => "/")
    relative == "Manifest.toml" && return true
    relative == ".git" && return true
    for prefix in ("docs/build", "docs/deploydocs")
        relative == prefix && return true
        startswith(relative, prefix * "/") && return true
    end
    return false
end

function _ownership_files(root::AbstractString)
    files = String[]
    for (dir, dirs, names) in walkdir(root)
        filter!(dirname -> !_ownership_excluded(joinpath(dir, dirname), root), dirs)
        for name in names
            path = joinpath(dir, name)
            _ownership_excluded(path, root) && continue
            push!(files, path)
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
        relative = relpath(path, root)
        for (line_number, line) in enumerate(eachline(IOBuffer(text)))
            if occursin(_FORMER_GITHUB_URL, line) || occursin(_FORMER_PAGES_URL, line)
                push!(hits, "$relative:$line_number")
            end
        end
    end
    @test isempty(hits)

    index = read(joinpath(root, "docs", "src", "index.md"), String)
    @test occursin("https://github.com/rmems/DendriteTrader.jl/actions/workflows/ci.yml", index)
    @test occursin("https://rmems.github.io/DendriteTrader.jl", index)
    install_commands =
        [strip(line) for line in split(index, '\n') if startswith(strip(line), "] add ")]
    @test install_commands == ["] add https://github.com/rmems/DendriteTrader.jl"]
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
