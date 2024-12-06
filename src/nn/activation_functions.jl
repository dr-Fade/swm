using Lux, Random, LinearAlgebra

variable_power(α, β, c) = u -> begin
    if u < -c^(1/(β-1))
        -(β-1) * c^(β/(β-1)) / β - (-u)^β / β
    elseif u ≤ c^(1/(α-1))
        c*u
    else
        (α-1) * c^(α/(α-1)) / α + u^α / α
    end
end

# α == β == c == 0.5
sqrt_activation(u) = begin
    if u < -4
        2 - 2*sqrt(-u)
    elseif u ≤ 4
        u / 2
    else
        -2 + 2sqrt(u)
    end
end

abs_square(params) = x -> x*abs(x)

leaky_tanh(; a=1f0, b=1f0, c=0.001f0) = x -> a*tanh_fast(b*x) + c*x

struct VariablePowerActivationFunction <: Lux.AbstractLuxLayer
end

function (f::VariablePowerActivationFunction)(u, ps, st)
    α, β = ps.parameters
    return variable_power(α, β, 1f0).(u), st
end

Lux.initialparameters(rng::AbstractRNG, f::VariablePowerActivationFunction) = (
    parameters = [1f0, 1f0],
)

Lux.initialstates(rng::AbstractRNG, f::VariablePowerActivationFunction) = NamedTuple()
