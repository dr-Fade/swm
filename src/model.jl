using Optimisers, Lux

include("sound_utils/frames.jl")

# the model takes MFCC as input and produces full spectrum as output

function loss_function(model, ps, st, data)
    xs, ys = data
    y_, st = model(xs[1], ps, st)
    loss = sum(abs2, y_ .- ys[1])
    for x ∈ xs[2:end]
        y_, st = model(xs[i], ps, st)
        loss += sum(abs2, y_ .- ys[i])
    end
    return loss, st, ()
end

function train(model, dataloader, epochs)
    rng = Random.default_rng()
    zygote = Lux.Training.ZygoteVJP()
    tstate = Lux.Training.TrainState(rng, model, Optimisers.ADAM(0.001f0));
    @time for epoch ∈ 1:epochs
        for batch ∈ dataloader
            grads, loss, stats, tstate = Lux.Training.compute_gradients(
                zygote,
                loss_function,
                batch,
                tstate
            )
            tstate = Lux.Training.apply_gradients(tstate, grads)
            Lux.@set! tstate.states = Lux.update_state(tstate.states, :carry, nothing)
        end
    end
    return tstate.parameters, tstate.states
end