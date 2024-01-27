using Optimisers, Zygote, Lux, MLUtils

function time_series_to_latent(encoder, ps, st, series::Vector, chunk_size::Int)
    return vcat([
        begin
            latent, st_encoder = encoder(chunk, ps, st)
            latent'
        end
        for chunk ∈ chunk_data([series;;], chunk_size)
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

function loss(model, ps, st, data)
    xs, ys = data
    pred_ys, st = model(xs, ps, st)
    loss = sum(abs2, ys .- pred_ys)
    return loss, st, pred_ys
end

function chunk_data(data::Matrix, chunk_size::Int; step::Int = 1)
    return [
        view(data, i:i+chunk_size-1, :)
        for i ∈ 1:step:size(data)[1]-chunk_size
    ]
end

function init_training_state(model, ps, st, optimiser)
    rng = Random.default_rng()
    tstate = Lux.Training.TrainState(rng, model, optimiser)
    if !isnothing(ps)
        Lux.@set! tstate.parameters = ps
    end
    if !isnothing(st)
        Lux.@set! tstate.states = st
    end
    return tstate
end

function train(
    model,
    data::DataLoader,
    loss_function;
    ps = nothing,
    st = nothing,
    batch_cb = nothing,
    epoch_cb = nothing,
    init_state = nothing,
    epochs = 1,
    optimiser = Optimisers.Adam()
)
    tstate = init_training_state(model, ps, st, optimiser)
    zygote = Lux.Training.AutoZygote()
    for epoch ∈ 1:epochs
        for batch ∈ data
            try
                if !isnothing(init_state)
                    Lux.@set! tstate.states = init_state(batch, ps, st)
                end
                grads, loss, stats, tstate = Lux.Training.compute_gradients(zygote, loss_function, batch, tstate)
                tstate = Lux.Training.apply_gradients(tstate, grads)
                if !isnothing(batch_cb)
                    batch_cb(loss_function, tstate.parameters, tstate.states)
                end
            finally
                Lux.@set! tstate.states = Lux.update_state(tstate.states, :carry, nothing)
            end
        end

        if !isnothing(epoch_cb)
            epoch_cb(loss_function, epoch, tstate.parameters, tstate.states)
        end
    end

    return tstate.parameters, tstate.states
end
