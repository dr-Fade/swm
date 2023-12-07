using MLUtils, ChaosTools, DynamicalSystems

function generate_lorenz_trajectory(; T = 1, ρ = 28, Δt = 0.01)
    u0 = [100rand(), 100rand(), 100rand()]
    lorenz = DynamicalSystems.Systems.lorenz(u0; σ = 10.0, ρ = ρ, β = 8/3)
    trajectory, t = DynamicalSystems.trajectory(lorenz, T; Ttr = 10, Δt = Δt)
    return trajectory
end

generate_lorenz_data_point(N; ρ = 28, d = 3, τ = 1, Δt = 0.1) = () -> begin
    T = (N + (d-1)) * Δt
    u = generate_lorenz_trajectory(; T = T, ρ = ρ, Δt = Δt)

    x = u[:,1]

    xs = [x[1:d];;]
    ys = [x[d:end];;]

    return xs, ys
end

function generate_lorenz_first_coordinate(N; ρ = 28, Δt = 0.1)
    T = N * Δt
    u = generate_lorenz_trajectory(; T = T, ρ = ρ, Δt = Δt)
    return [u[1:N,1] |> Vector{Float32};;]
end

sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

generate_sine_data_point(N, f0=220; harmonics = 1, d = 2, τ = 1, Δt = 0.00001) = () -> begin
    T = (N + (d-1)*τ) * Δt
    t = 0:Δt:T
    ϕ = 10rand()

    x = reduce(+, [sine_wave(f, 1, ϕ).(t) for f ∈ (1:harmonics) .* f0])

    xs = [x[1:d];;]
    ys = [x[d:end];;]

    return xs, ys
end

parametrized_sin_data_point_generator(N, f0=220; harmonics = 1, d = 2, τ = 1, Δt = 0.00001) = () -> begin
    params = [f0]

    Δt = 1 / sample_rate
    T = (N + (d-1)*τ) * Δt
    t = 0:Δt:T
    ϕ = 10rand()

    x = reduce(+, [sine_wave(f, 1, ϕ).(t) for f ∈ (1:harmonics) .* f0])

    xs = [params; x[1:d];;]
    ys = [x[d:end];;]

    return xs, ys
end

function rnn_trajectory(model, ps, st, u0, N)
    n = size(u0)[1]
    traj = u0
    for i ∈ 1:(N-1)
        y, st = model(traj[end-n+1:end,:], ps, st)
        traj = vcat(traj, y)
    end
    return traj[n:end,:], st
end

function latent_rnn_trajectory(model, ps, st, u0, N)
    traj = u0
    ly = latent_trajectory(model, traj, ps, st).u[1]'
    latent_traj = ly
    for i ∈ 1:N
        y, st = model(traj, ps, st)
        traj = vcat(traj[2:end], y)
        ly = vcat(latent_trajectory(model, traj, ps, st).u'...)
        latent_traj = vcat(latent_traj, ly)
    end
    return latent_traj, st
end

function generate_data(generator; batches = 200)
    data = []

    data = foldl(
        (cur, _) -> begin
            xs, ys = generator()
            if isempty(cur)
                (xs, ys)
            else
                cur_xs, cur_ys = cur
                (hcat(cur_xs, xs), hcat(cur_ys, ys))
            end
        end,
        1:batches;
        init = data
    )

    return data
end
