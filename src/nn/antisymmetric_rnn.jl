using Lux, Random, LinearAlgebra

struct AntisymmetricRNN{A, B, W, S} <: Lux.AbstractExplicitLayer
    n::Int
    activation::A
    init_bias::B
    init_weight::W
    init_state::S
    function AntisymmetricRNN(n, activation=tanh; init_bias=Lux.zeros32, init_weight=Lux.glorot_uniform, init_state=Lux.zeros32)
        new{typeof(activation), typeof(init_bias), typeof(init_weight), typeof(init_state)}(
            n, activation, init_bias, init_weight, init_state
        )
    end
end

Lux.initialparameters(rng::AbstractRNG, f::AntisymmetricRNN) = (
    W = f.init_weight(rng, f.n, f.n),
    Vh = f.init_weight(rng, f.n, f.n),
    bh = f.init_bias(rng, f.n),
    Vz = f.init_weight(rng, f.n, f.n),
    bz = f.init_bias(rng, f.n),
    γ = abs.(rand(rng)),
    ε = abs.(rand(rng))
)

Lux.initialstates(rng::AbstractRNG, f::AntisymmetricRNN) = (
    rng = Lux.replicate(rng),
)

function (f::AntisymmetricRNN)(xs::AbstractVector, ps, st::NamedTuple)
    return f([xs;;], ps, st)
end

function (f::AntisymmetricRNN)(xs::AbstractMatrix, ps, st::NamedTuple)
    hidden_state = repeat(f.init_state(st.rng, f.n), 1, size(xs, 2))
    return f((xs, (hidden_state,)), ps, st)
end

function (f::AntisymmetricRNN)(xs::Union{CUDA.StridedSubCuArray, CuArray}, ps, st::NamedTuple)
    hidden_state = repeat(f.init_state(st.rng, f.n), 1, size(xs, 2)) |> gpu
    return f((xs, (hidden_state,)), ps, st)
end

function (f::AntisymmetricRNN)(
    (x, (hidden_state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
    ps,
    st::NamedTuple
)
    A = (ps[:W] - ps[:W]' - (abs(ps.γ)*I(f.n)))*hidden_state
    z = Lux.σ.(A + ps[:Vz]*x .+ ps[:bz])
    d = z .* f.activation.(A + ps.Vh*x .+ ps.bh)
    h_new = hidden_state + abs(ps.ε)*d
    return (h_new, (h_new,)), st
end

function (f::AntisymmetricRNN)(
    (x, (hidden_state,))::Tuple{<:Union{CUDA.StridedSubCuArray, CuArray}, Tuple{<:Union{CUDA.StridedSubCuArray, CuArray}}},
    ps,
    st::NamedTuple
)
    A = (ps[:W] - ps[:W]' - abs(ps.γ) * (Diagonal(ones(f.n)) |> gpu))*hidden_state
    z = Lux.σ.(A + ps[:Vz]*x .+ ps[:bz])
    d = z .* f.activation.(A + ps.Vh*x .+ ps.bh)
    h_new = hidden_state + abs(ps.ε)*d
    return (h_new, (h_new,)), st
end
