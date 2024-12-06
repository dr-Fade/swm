using BSON: @load, @save
using DynamicalSystems, DifferentialEquations, ComponentArrays, DiffEqFlux, Random

struct hNODE <: Lux.AbstractLuxContainerLayer{(:encoder, :decoder, :control)}
    encoder::Lux.AbstractLuxLayer
    decoder::Lux.AbstractLuxLayer
    control::Lux.AbstractLuxLayer
    ode::Lux.AbstractLuxLayer
    ode_axes::Tuple

    function hNODE(
        encoder::Lux.AbstractLuxLayer,
        decoder::Lux.AbstractLuxLayer,
        control::Lux.AbstractLuxLayer,
        ode::Lux.AbstractLuxLayer
    )
        ode_axes = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> getaxes
        return new(encoder, decoder, control, ode, ode_axes)
    end
end

Lux.initialstates(rng::AbstractRNG, m::hNODE) = (
    rng = rng,
    control = Lux.initialstates(rng, m.control),
    encoder = Lux.initialstates(rng, m.encoder),
    decoder = Lux.initialstates(rng, m.decoder),
    Δt = 0.1f0,
    T = 1f0,
    ode = Lux.initialstates(rng, m.ode)
)

get_encoder(m::hNODE, ps, st) = begin
    m.encoder, ps.encoder, st.encoder
end

get_decoder(m::hNODE, ps, st) = begin
    m.decoder, ps.decoder, st.decoder
end

get_control(m::hNODE, ps, st) = begin
    m.control, ps.control, st.control
end

# first stage - embed into the latent space
function (m::hNODE)(xs::AbstractMatrix, ps, st)
    u0s, st_encoder = m.encoder(xs, ps.encoder, st.encoder)
    return m((xs, (u0s,)), ps, (st..., encoder = st_encoder))
end

# second stage - extract the feature vector
function (m::hNODE)(
    (xs, (u0s,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
    ps,
    st::NamedTuple
)
    controls, st_control = m.control(xs, ps.control, st.control)
    return m((xs, (u0s, controls)), ps, (st..., control = st_control))
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

# third stage - use the feature vector to get the control for ode and integrate it using the embedded sound as u0
function (m::hNODE)(
    (_, (u0s, controls))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
    ps,
    st::NamedTuple
)
    tspan = (0.0f0, st.T)
    node = NeuralODE(m.ode, tspan, Tsit5(), saveat=0f0:st.Δt:st.T, save_on=true, save_start=true, save_end=true, verbose=false)

    ys, uns = [
        begin
            ode_params = ComponentArray(control, m.ode_axes)
            ode_solution = node(u0, ode_params, st.ode)[1].u

            decoded_trajectory = m.decoder(reduce(hcat, ode_solution[1:end-1]), ps.decoder, st.decoder)[1]
            new_u0s = ode_solution[end]

            decoded_trajectory', new_u0s
        end
        for (control, u0) ∈ zip(
            eachslice(controls; dims = batch_dimension(controls)),
            eachslice(u0s; dims = batch_dimension(u0s))
        )
    ] |> unzip

    ys = cat(ys...; dims=batch_dimension(ys[1])+1)
    uns = cat(uns...; dims=batch_dimension(uns[1])+1)

    return (ys, (uns,)), st
end

batch_dimension(xs) = length(size(xs))
