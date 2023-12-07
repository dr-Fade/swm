using Optimisers, Zygote, Lux, MLUtils

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
    cb = nothing,
    init_state = nothing,
    epochs = 1,
    optimiser = Optimisers.Adam()
)
    tstate = init_training_state(model, ps, st, optimiser)
    zygote = Lux.Training.ZygoteVJP()
    for epoch ∈ 1:epochs
        for batch ∈ data
            if !isnothing(init_state)
                Lux.@set! tstate.states = init_state(batch, ps, st)
            end
            grads, loss, stats, tstate = Lux.Training.compute_gradients(zygote, loss_function, batch, tstate)
            tstate = Lux.Training.apply_gradients(tstate, grads)
            Lux.@set! tstate.states = Lux.update_state(tstate.states, :carry, nothing)
        end

        if !isnothing(cb)
            cb(epoch, tstate.parameters, tstate.states)
        end
    end

    return tstate.parameters, tstate.states
end
