using Lux, Random, LinearAlgebra

struct AntisymmetricBlock <: Lux.AbstractExplicitLayer
    n
    σ
    function AntisymmetricBlock(n, σ)
        new(n, σ)
    end
end

function (f::AntisymmetricBlock)(u, p, st)
    return (
        f.σ.((p[:W] - p[:W]' + I(f.n) .* p[:L])*u .+ p[:b]),
        st
    )
end

Lux.initialparameters(rng::AbstractRNG, f::AntisymmetricBlock) = (
    W = rand(rng, f.n, f.n),
    L = zeros(f.n),
    b = zeros(f.n)
)

Lux.initialstates(rng::AbstractRNG, f::AntisymmetricBlock) = NamedTuple()
