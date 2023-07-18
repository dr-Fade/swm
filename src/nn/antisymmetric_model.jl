using Lux, Random, LinearAlgebra

struct AntisymmetricBlock <: Lux.AbstractExplicitLayer
    n
    σ
    init_σ_params
    function AntisymmetricBlock(n, σ, init_σ_params)
        new(n, σ, init_σ_params)
    end
end

function (f::AntisymmetricBlock)(u, p, st)
    applied_antisymmetric = (p[:W] - p[:W]')*u + p[:b]
    return (
        [f.σ(p[:σps][i,:])(applied_antisymmetric[i]) for i ∈ 1:f.n] + p[:L]*u,
        st
    )
end

Lux.initialparameters(rng::AbstractRNG, f::AntisymmetricBlock) = (
    W = rand(rng, f.n, f.n),
    L = zeros(f.n, f.n),
    b = zeros(f.n),
    σps = [f.init_σ_params[i] for j ∈ 1:f.n, i ∈ 1:length(f.init_σ_params)]
)

Lux.initialstates(rng::AbstractRNG, f::AntisymmetricBlock) = NamedTuple()

Lux.parameterlength(f::AntisymmetricBlock) = f.n*(2*f.n + 1) + length(f.init_σ_params)

Lux.statelength(f::AntisymmetricBlock) = 0
