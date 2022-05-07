using DifferentialEquations, DiffEqFlux, Plots, DynamicalSystems, Flux

function approximate_with_node(
    time_series,
    dimensions,
    lag;
    tspan = (0f0, 10f0),
    N = 10,
    group_size = 3,
    continuity_term = 200
)
    target = hcat(embed(time_series, dimensions, lag)...) |> Matrix
    dimensions, datasize = size(target)
    if dimensions >= datasize
        error("The number of dimensions cannot be greater than the number of points.")
    end
    tsteps = range(tspan[begin], tspan[end], length = datasize)

    # TODO: parametrize depth, widths, and activation functions
    nn = FastChain(
        FastDense(dimensions, N, x -> x .^ 1/3),
        FastDense(N, N, x -> x .^ 1/3),
        FastDense(N, N, x -> x .^ 1/3),
        FastDense(N, dimensions)
    )

    u0 = target[:,begin]
    p_init = initial_params(nn)
    prob_node = ODEProblem((u,p,t)->nn(u,p), u0, tspan, p_init)

    callback = function (p, l, preds)
        display(l)
        pred_tspan = (tspan[begin], 2*tspan[end])
        pred_tsteps = range(pred_tspan[begin], pred_tspan[end], length = datasize)
        pred = Array(
            solve(
                ODEProblem((u,p,t)->nn(u,p), u0, pred_tspan, p),
                Tsit5(),
                saveat = pred_tsteps
            )
        )
        plt = scatter(tsteps[begin:datasize], target[1,begin:datasize], label = "Data")
        scatter!(plt, pred_tsteps, pred[1,:], label = "Prediction")
        display(plt)
        return false
    end

    function loss_function(data, pred)
        return sum(abs2, data - pred)
    end

    function loss_multiple_shooting(p)
        return multiple_shoot(
            p,
            target,
            tsteps,
            prob_node,
            loss_function,
            Tsit5(),
            group_size;
            continuity_term
        )
    end

    return DiffEqFlux.sciml_train(loss_multiple_shooting, p_init, cb = callback)
end

function approximate_sin_sum()
    tspan = (0f0, 25f0)
    datasize = 100
    tsteps = range(tspan[begin], tspan[end], length = datasize)
    # base tone and perfect fifth for ~2 cycles
    target_data = sin.(tsteps) .+ sin.(3 .* tsteps ./ 2) ./ 2
    return approximate_with_node(target_data, 5, 6; tspan)
end

function approximate_polynomial()
    tspan = (0f0, 5f0)
    tsteps = range(tspan[begin], tspan[end], length = datasize)
    target_data = (x -> (x-1)*(x-2)*(x-3)*(x-4)).(tsteps)
    return approximate_with_node(target_data, 1, 6; tspan)
end

approximate_sin_sum()