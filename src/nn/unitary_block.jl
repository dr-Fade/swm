using Lux, Random, LinearAlgebra

struct UnitaryBlock <: Lux.AbstractExplicitLayer
    in_dims::Integer
    out_dims::Integer
    σ::Function
    function UnitaryBlock(in_dims, out_dims, σ=identity)
        new(in_dims, out_dims, σ)
    end
end

function (f::UnitaryBlock)(u, ps, st)
    return f.σ.(normalized_w(ps) * u), st
end

normalized_w(ps) = vcat([row' / norm(row) for row ∈ eachrow(ps.W)]...)

Lux.initialparameters(rng::AbstractRNG, f::UnitaryBlock) = (
    W = rand(rng, Float32, f.out_dims, f.in_dims),
)

Lux.initialstates(rng::AbstractRNG, f::UnitaryBlock) = NamedTuple()
