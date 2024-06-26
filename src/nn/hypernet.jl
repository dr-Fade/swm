using Lux, Random, LinearAlgebra

struct HypernetBlock <: Lux.AbstractExplicitContainerLayer
    control::Lux.AbstractExplicitLayer
    model::Lux.AbstractExplicitLayer
    model_axes::Tuple
    context_size::Int
end

function HypernetBlock(control::Lux.AbstractExplicitLayer, model::Lux.AbstractExplicitLayer, model_axes::Tuple; context_size::Int=0)
    return HypernetBlock(control, model, model_axes, context_size)
end

function (hb::HypernetBlock)(x, ps, st)
    params, st_control = if isnothing(st[:context])
        ps.model, st.control
    else
        control_flat, st_control = hb.control(st[:context], ps.control, st.control) 
        control = ComponentArray(vec(control_flat), hb.model_axes)
        params_with_control = ComponentArray(ps.model) + control
        params_with_control, st_control
    end

    y, st_model = hb.model(x, params, st.model)

    return y, (model=st_model, control=st_control)
end

Lux.initialparameters(rng::AbstractRNG, f::HypernetBlock) = (
    control = Lux.initialparameters(rng, f.control),
    model = Lux.initialparameters(rng, f.model)
)

Lux.initialstates(rng::AbstractRNG, hb::HypernetBlock) = (
    control = Lux.initialstates(rng, hb.control),
    model = Lux.initialstates(rng, hb.model),
    context = nothing
)

