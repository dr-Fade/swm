using DifferentialEquations, DiffEqFlux, Plots, DynamicalSystems, Flux

struct HamiltonianModel{F}
    initial_params::F
    function HamiltonianModel()
        Y = zeros(20)
        Y[3] = 1
        Y[6] = -1
        initial_params() = Y
        new{typeof(initial_params)}(initial_params)
    end

end

# Overload call, so the object can be used as a function
function (m::HamiltonianModel)(u,p=m.initial_params())
    x, y, z = u
    c1, a11, a12, a13,
    c2, a21, a22, a23,
    c3, a31, a32, a33,
    p, s, q, d, h, e, f, r = p
    return [
        c1 + a11*x + a12*y + a13*z + p*x*y + s*y^2 + q*x*z + d*y*z + h * z^2
        c2 + a21*x + a22*y + a23*z - p*x^2 - s*x*y + e*x*z + f*y*z + r*z^2
        c3 + a31*x + a32*y + a33*z - q*x^2 - (d+e)*x*y - f*y^2 - h*x*z - r*y*z
    ]
end

DiffEqFlux.initial_params(f::HamiltonianModel) = f.initial_params()
DiffEqFlux.paramlength(f::HamiltonianModel) = 20
