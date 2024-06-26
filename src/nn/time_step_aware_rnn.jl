using Lux, Random, LinearAlgebra

struct TimeStepAwareRNN <: Lux.AbstractRecurrentCell{true, false}
    in_dims::Integer
    out_dims::Integer
    dt::Float32
    σ::Function
    function TimeStepAwareRNN(in_dims, out_dims, σ=tanh; dt=1f0)
        new(in_dims, out_dims, dt, σ)
    end
end

function (f::TimeStepAwareRNN)(x::AbstractMatrix, ps, st::NamedTuple)
    return f((x, (zeros32(f.out_dims, size(x)[2]),)), ps, st)
end

function (f::TimeStepAwareRNN)((x, (hidden,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}}, ps, st::NamedTuple)
    y = ps.Wi*x .+
        (ps.Wh / f.dt)*hidden .+
        ps.b
    y = f.σ.(y)
    return (y, (y,)), st
end

Lux.initialparameters(rng::AbstractRNG, f::TimeStepAwareRNN) = (
    Wi = glorot_normal(rng, f.out_dims, f.in_dims),
    Wh = glorot_normal(rng, f.out_dims, f.out_dims),
    b = zeros32(f.out_dims)
)

Lux.initialstates(rng::AbstractRNG, f::TimeStepAwareRNN) = NamedTuple()
