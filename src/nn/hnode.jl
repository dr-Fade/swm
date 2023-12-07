using Lux, ComponentArrays, DiffEqFlux, DifferentialEquations, Random, ChaosTools
include("antisymmetric_model.jl")
include("hypernet.jl")
include("activation_functions.jl")

struct hNODE <: Lux.AbstractExplicitLayer
    hb::Lux.AbstractExplicitLayer
    encoder::Lux.AbstractExplicitLayer
    decoder::Lux.AbstractExplicitLayer
    in_n::Int
    latent_n::Int
    out_n::Int
end


function conv_layer(in_n, kernel_size; σ=tanh)
    return Lux.Chain(
        x -> reshape(x, (in_n, 1, :)),
        Lux.Conv((kernel_size,), 1=>1, σ),
        x -> x[:,1,:]
    ), in_n - kernel_size + 1
end


function hNODE(in_n::Int, latent_n::Int, out_n::Int; context_n::Int = 0, activation=identity)
    hidden_n = in_n + latent_n + out_n
    encoder = Lux.Chain(
        Lux.Dense(in_n, hidden_n, tanh),
        Lux.Dense(hidden_n, latent_n)
    )

    decoder = Lux.Chain(
        Lux.Dense(latent_n, out_n)
    )

    # activation = variable_power(0.5, 0.3, 5)
    # activation = cbrt
    # activation = tanh
    # activation = relu
    ode = AntisymmetricBlock(latent_n, activation)

    ode_axes = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> getaxes

    control = if context_n > 0
        ode_params_n = Lux.parameterlength(ode)
        control_conv, control_conv_out = conv_layer(ode_params_n, ode_params_n÷10; σ=tanh)
        Lux.Chain(
            Lux.Dense(latent_n, ode_params_n, tanh),
            control_conv,
            Lux.Dense(control_conv_out, ode_params_n)
        )
    else
        Lux.Chain()
    end

    # encoder_conv, encoder_conv_out = conv_layer(in_n, in_n÷latent_n; σ=tanh)
    # encoder = Lux.Chain(
    #     Lux.Dense(in_n, in_n, tanh),
    #     encoder_conv,
    #     Lux.Dense(encoder_conv_out, latent_n)
    # )

    # decoder_conv, decoder_conv_out = conv_layer(latent_n, latent_n÷(out_n+1); σ=tanh)
    # decoder = Lux.Chain(
    #     decoder_conv,
    #     Lux.Dense(decoder_conv_out, out_n)
    # )

    # activation = variable_power(0.5, 1.5, 5)
    # ode = AntisymmetricBlock(latent_n, activation)

    # ode_axes = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> getaxes

    # control = if context_n > 0
    #     ode_params_n = Lux.parameterlength(ode)
    #     control_conv, control_conv_out = conv_layer(ode_params_n, ode_params_n÷10; σ=tanh)
    #     Lux.Chain(
    #         Lux.Dense(latent_n, ode_params_n, tanh),
    #         control_conv,
    #         Lux.Dense(control_conv_out, ode_params_n)
    #     )
    # else
    #     Lux.Chain()
    # end

    hb = HypernetBlock(control, ode, ode_axes)

    return hNODE(hb, encoder, decoder, in_n, latent_n, out_n)
end

Lux.initialparameters(rng::AbstractRNG, hnode::hNODE) = (
    hb = Lux.initialparameters(rng, hnode.hb),
    encoder = Lux.initialparameters(rng, hnode.encoder),
    decoder = Lux.initialparameters(rng, hnode.decoder)
)

Lux.initialstates(rng::AbstractRNG, hnode::hNODE) = (
    hb = Lux.initialstates(rng, hnode.hb),
    encoder = Lux.initialstates(rng, hnode.encoder),
    decoder = Lux.initialstates(rng, hnode.decoder),
    saveat = nothing,
    save_on = false,
    save_start = false,
    save_end = true,
    T = 1.0
)

function (hnode::hNODE)(x::NamedTuple, ps, st::NamedTuple)
    return hnode(x[:context], ps, st)
end

function (hnode::hNODE)(xs::AbstractVector, ps, st::NamedTuple)
    return hnode([xs;;], ps, st)
end

function (hnode::hNODE)(
    x::AbstractMatrix,
    ps,
    st::NamedTuple
)
    u0, st_encoder = hnode.encoder(x, ps.encoder, st.encoder)

    tspan = (0.0, st.T)
    node = NeuralODE(hnode.hb, tspan, Tsit5(), saveat=st[:saveat], save_on=st[:save_on], save_start=st[:save_start], save_end=st[:save_end], verbose = false)

    solution, st_hb = node(u0, ComponentArray(ps.hb), st.hb)

    st_decoder = st[:decoder]
    u_new = vcat([
        begin
            y, st_decoder = hnode.decoder(u, ps.decoder, st.decoder)
            y
        end
        for u ∈ solution.u
    ]...)

    return u_new, (st..., hb = st_hb, encoder = st_encoder, decoder = st_decoder)
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