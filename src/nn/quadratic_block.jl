using Lux, Random, LinearAlgebra

struct QuadraticBlock <: Lux.AbstractExplicitLayer
    n::Integer
    σ::Function
    function QuadraticBlock(n, σ=identity)
        new(n, σ)
    end
end

function (f::QuadraticBlock)(u, ps, st)
    return f.σ.(u .* ps.W * u), st
end

Lux.initialparameters(rng::AbstractRNG, f::QuadraticBlock) = (
    W = rand(rng, Float32, f.n, f.n),
)

Lux.initialstates(rng::AbstractRNG, f::QuadraticBlock) = NamedTuple()
