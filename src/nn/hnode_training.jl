include("hnode.jl")
include("training_utils.jl")


function time_series_to_latent(encoder::Lux.AbstractExplicitLayer, ps, st, series::Matrix, chunk_size::Int)
    return vcat([
        begin
            latent, st_encoder = encoder(chunk, ps, st)
            latent'
        end
        for chunk ∈ chunk_data(series, chunk_size)
    ]...)
end

function latent_to_time_series(decoder, ps, st, latent_points::Matrix)
    return vcat([
        begin
            lp = latent_points[i,:]
            tsp, st_decoder = decoder(lp, ps, st)
            tsp
        end
        for i ∈ 1:size(latent_points)[1]
    ]'...)
end

function train_encoders(hnode::hNODE, ps, st::NamedTuple, input_n, series; batchsize = 1, chunk_step = 1, epochs = 1, optimiser = Optimisers.Adam())
    # model
    model = Lux.Chain(;
        encoder = hnode.encoder,
        decoder = hnode.decoder,
        disable_optimizations = true
    )
    model_ps = (encoder = ps.encoder, decoder = ps.decoder)
    model_st = (encoder = st.encoder, decoder = st.decoder)

    # data
    xs = hcat(chunk_data(series, input_n; step=chunk_step)...)
    ys = xs[end,:]
    train_data = (xs, ys)
    loader = DataLoader(train_data; batchsize = batchsize, shuffle = true)

    #training
    new_ps, new_st = train(model, loader, loss; ps = model_ps, st = model_st, epochs = epochs, optimiser = optimiser)

    return (ps..., encoder = new_ps.encoder, decoder = new_ps.decoder),
            (st..., encoder = new_st.encoder, decoder = new_st.decoder)
end

function train_nested_ode(
    hnode::hNODE,
    ps,
    st::NamedTuple,
    Δt,
    series;
    integration_length = 5,
    batchsize = 1,
    epochs = 100,
    cb = nothing,
    optimizer = Optimisers.Adam(0.001)
)
    # model
    T = integration_length * Δt
    tspan = (0.0, T)
    node = NeuralODE(hnode.hb, tspan, Tsit5(), saveat=Δt:Δt:T, save_on=true, save_start=false, save_end=true, verbose = true)
    hb_ps = ps.hb
    hb_st = st.hb

    #data
    target_trajectory = time_series_to_latent(hnode.encoder, ps.encoder, st.encoder, series, hnode.in_n)
    xs = target_trajectory[1:end-integration_length,:]' # ode_dimension x batch
    ys = cat([target_trajectory[i+1:i+integration_length,:] for i ∈ 1:size(xs)[2]]...; dims=3) # point_n x ode_dimension x batch
    loader = DataLoader((xs, ys); batchsize = batchsize, shuffle = true)

    #training
    function integrated_loss(model, ps, st, data)
        xs, ys = data
        solution, st_hb = model(xs, ComponentArray(ps), st) # point_n [ode_dimension x batch]
        loss = 0.0
        for i ∈ 1:length(solution.u)
            u = solution.u[i]
            y = ys[i,:,:]
            loss += sum(abs2, u .- y)
        end
        return loss, st_hb, ()
    end

    new_ps, new_st = train(node, loader, integrated_loss; ps = hb_ps, st = hb_st, epoch_cb = cb, epochs = epochs, optimiser = optimizer)

    return (ps..., hb = new_ps), (st..., hb = new_st)
end
