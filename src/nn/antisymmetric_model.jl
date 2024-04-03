using Lux, Random, LinearAlgebra

struct AntisymmetricBlock <: Lux.AbstractExplicitLayer
    n::Integer
    σ::Function
    data_type::DataType
    function AntisymmetricBlock(n, σ=tanh; data_type = Float32)
        new(n, σ, data_type)
    end
end

Base.to_indices(x::ComponentArray, i::Tuple{Any}) = i

function (f::AntisymmetricBlock)(u, p, st)
    return (
        f.σ.((p[:W] - p[:W]' + Diagonal(p[:L]))*u .+ p[:b]),
        st
    )
end

Lux.initialparameters(rng::AbstractRNG, f::AntisymmetricBlock) = (
    W = rand(rng, f.data_type, f.n, f.n),
    L = zeros(f.data_type, f.n),
    b = zeros(f.data_type, f.n)
)

Lux.initialstates(rng::AbstractRNG, f::AntisymmetricBlock) = NamedTuple()
