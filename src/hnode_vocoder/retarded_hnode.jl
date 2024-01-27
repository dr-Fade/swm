using Lux, ComponentArrays, DiffEqFlux, DifferentialEquations, Random, ChaosTools
include("antisymmetric_model.jl")
include("hypernet.jl")
include("activation_functions.jl")

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

struct hNODE <: Lux.AbstractExplicitContainerLayer{(:control, :encoder, :decoder)}
    control::Lux.AbstractExplicitLayer
    encoder::Lux.AbstractExplicitLayer
    decoder::Lux.AbstractExplicitLayer
    ode::Lux.AbstractExplicitLayer
    ode_axes::Tuple
    add_noise::Bool

    function hNODE(control, encoder, decoder, ode; add_noise = false)
        ode_axes = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> getaxes
        return new(control, encoder, decoder, ode, ode_axes, add_noise)
    end

end

Lux.initialstates(rng::AbstractRNG, hnode::hNODE) = (
    rng = rng,
    control = Lux.initialstates(rng, hnode.control),
    encoder = Lux.initialstates(rng, hnode.encoder),
    decoder = Lux.initialstates(rng, hnode.decoder),
    ode = Lux.initialstates(rng, hnode.ode),
    saveat = nothing,
    save_on = false,
    save_start = false,
    save_end = true,
    T = 1.0
)

function (hnode::hNODE)(xs::AbstractVector, ps, st::NamedTuple)
    return hnode([xs;;], ps, st)
end

# the regular mode: both the encoder and the control take the time series as the input
function (hnode::hNODE)(
    xs::AbstractMatrix,
    ps,
    st::NamedTuple
)
    u0, _ = hnode.encoder(xs, ps.encoder, st.encoder)
    return hnode((xs, (u0,)), ps, st)
end

# the alternative mode: a separate set of data can be passed for the controller.
# this is done to allow non-NN related operations that might not be supported by the optimizer or are not required to be recomputed every time
function (hnode::hNODE)(
    (encoder_xs, control_xs)::Tuple{<:AbstractMatrix, <:AbstractMatrix},
    ps,
    st::NamedTuple
)
    u0, _ = hnode.encoder(encoder_xs, ps.encoder, st.encoder)
    return hnode((control_xs, (u0,)), ps, st)
end

function (hnode::hNODE)(
    (xs, (u0,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
    ps,
    st::NamedTuple
)
    tspan = (0.0f0, st.T)
    node = NeuralODE(hnode.ode, tspan, Tsit5(), saveat=st[:saveat] |> Lux.cpu, save_on=st[:save_on], save_start=st[:save_start], save_end=st[:save_end], verbose = false)
    control, _ = hnode.control(xs, ps.control, st.control)

    ys, un = [
        begin
            ode_params = ComponentArray(view(control,:,i), hnode.ode_axes)
            ode_solution = node(view(u0,:,i), ode_params, st.ode)[1].u
            decoded_trajectory = hnode.decoder(reduce(hcat, ode_solution), ps.decoder, st.decoder)[1]
            new_u0 = ode_solution[:,end]

            final_trajectory = if hnode.add_noise
                noise_intensity = σ(control[end, i])
                noise = noise_intensity .* (2 * rand(st.rng, eltype(decoded_trajectory), size(decoded_trajectory)) .- 1)
                noise .+ decoded_trajectory
            else
                decoded_trajectory
            end
            
            final_trajectory', new_u0
        end
        for i ∈ 1:size(xs)[2]
    ] |> unzip

    return (ys, (reduce(hcat, un),)), st
end

function latent_trajectory(
    hnode::hNODE,
    x::AbstractMatrix,
    ps,
    st::NamedTuple
)
    u0, _ = hnode.encoder(x, ps.encoder, st.encoder)
    tspan = (0.0, st.T)

    node = NeuralODE(hnode.hb, tspan, Tsit5(), saveat=st[:saveat], save_on=st[:save_on], save_start=st[:save_start], save_end=st[:save_end], verbose = false)

    solution, _ = node(u0, ComponentArray(ps.hb), (st.hb..., context = x))

    return solution
end

function latent_lyaps(
    hnode::hNODE,
    x::AbstractMatrix,
    ps,
    st::NamedTuple
)
    u0, _ = hnode.encoder(x, ps.encoder, st.encoder)

    node_as_cds = ContinuousDynamicalSystem((x,_,_) -> hnode.hb(x, ComponentArray(ps.hb), (st.hb..., context = x))[1] |> SVector{3}, u0)

    return lyapunovspectrum(node_as_cds, 100)
end

function update_state(
    st::NamedTuple;
    saveat = [],
    save_on = false,
    save_start = false,
    save_end = true,
    T = nothing
)::NamedTuple

    Lux.@set! st = Lux.update_state(st, :saveat, saveat)
    Lux.@set! st = Lux.update_state(st, :save_on, save_on)
    Lux.@set! st = Lux.update_state(st, :save_start, save_start)
    Lux.@set! st = Lux.update_state(st, :save_end, save_end)
    Lux.@set! st = Lux.update_state(st, :T, T)

    return st
end

struct hNODERecurrent <: Lux.AbstractRecurrentCell{false, false}
    hnode::hNODE
end

Lux.initialparameters(rng::AbstractRNG, m::hNODERecurrent) = Lux.initialparameters(rng, m.hnode)
Lux.initialstates(rng::AbstractRNG, m::hNODERecurrent) = Lux.initialstates(rng, m.hnode)

(m::hNODERecurrent)(
    (encoder_xs, control_xs)::Tuple{<:AbstractMatrix, <:AbstractMatrix},
    ps,
    st::NamedTuple
) = m.hnode((encoder_xs, control_xs), ps, st)

(m::hNODERecurrent)(
    xs::AbstractMatrix,
    ps,
    st::NamedTuple
) = m.hnode(xs, ps, st)

(m::hNODERecurrent)(
    (xs, (u0,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
    ps,
    st::NamedTuple
) = m.hnode((xs, (u0,)), ps, st)
