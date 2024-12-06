using Optimisers, Zygote, Lux, MLUtils, ADTypes, Setfield

unixepoch() = datetime2unix(now()) |> round |> Int

function time_series_to_latent(encoder, ps, st, series::Vector, chunk_size::Int)
    return reduce(vcat, [
        begin
            latent, st_encoder = encoder(chunk, ps, st)
            latent'
        end
        for chunk ∈ chunk_data([series;;], chunk_size)
    ])
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
        for i ∈ 1:step:size(data)[1]-chunk_size+1
    ]
end

function init_training_state(model, ps, st, optimiser, distributed_backend=nothing)
    tstate = Lux.Training.TrainState(model, ps, st, isnothing(distributed_backend) ? optimiser : DistributedUtils.DistributedOptimizer(distributed_backend, optimiser))
    if !isnothing(distributed_backend)
        Setfield.@set! tstate.parameters = DistributedUtils.synchronize!!(distributed_backend, tstate.parameters)
        Setfield.@set! tstate.states = DistributedUtils.synchronize!!(distributed_backend, tstate.states)
        Setfield.@set! tstate.optimizer_state = DistributedUtils.synchronize!!(distributed_backend, tstate.optimizer_state)
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
    epochs = 1,
    optimiser = Optimisers.Adam(),
    states_to_clear = (:carry,),
    distributed_backend = nothing
)
    tstate = init_training_state(model, ps, st, optimiser, distributed_backend)
    zygote = ADTypes.AutoZygote()
    epoch = 1
    batch_N = length(data)
    while epoch <= epochs    
        for (batch_i, batch) ∈ enumerate(data)
            try
                grads, loss, stats, tstate = Lux.Training.compute_gradients(zygote, loss_function, batch, tstate)

                if any(x -> !isnothing(x) && isnan(x), ComponentArray(grads))
                    println("$epoch/$epochs ($batch_i/$batch_N): NaNs detected in gradients, not applying.")
                    continue
                    # println("$epoch/$epochs ($batch_i/$batch_N): Replacing NaNs in gradients with random values")
                    # grads = map(x -> isnothing(x) ? x : isnan(x) ? 0.0001f0*(rand(Float32)-0.5f0) : x, ComponentArray(grads))
                end
                prev_params = tstate.parameters
                tstate = Lux.Training.apply_gradients(tstate, grads)
                if any(isnan, ComponentArray(tstate.parameters))
                    error("NaN detected in the parameters!")
                end
            finally
                for state in states_to_clear
                    Setfield.@set! tstate.states = Lux.update_state(tstate.states, state, nothing)
                end
            end
            if !isnothing(batch_cb)
                batch_cb(loss_function, model, tstate.parameters, tstate.states, batch_i, batch_N, epoch)
            end
        end

        if !isnothing(epoch_cb) && epoch_cb(loss_function, epoch, epochs, model, tstate.parameters, tstate.states)
            break
        end

        epoch += 1
    end
    if !isnothing(distributed_backend)
        println("worker $(DistributedUtils.local_rank(distributed_backend)): done!")
    end
    return tstate.parameters, tstate.states
end
