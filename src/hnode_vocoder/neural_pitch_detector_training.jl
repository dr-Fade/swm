include("../nn/training_utils.jl")
include("../sound_utils/neural_pitch_detector.jl")

using ComponentArrays, LinearAlgebra

sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

function create_synthesized_training_data(model::NeuralPitchDetector, f0::Float32, A::Float32, ϕ::Float32;
    harmonics=1,
    β::Float32=0f0,
    N=nothing,
    window=ones32,
)
    r = range(0, 1, model.sample_rate)
    step = Int(model.sample_rate÷F0_CEIL)
    batches = length(r) ÷ step

    N = isnothing(N) ? batches÷2 : min(N, batches)
    batches = batches

    ϕ = f0 == 0f0 ? 0f0 : ϕ
    noise = 2*rand32(Random.default_rng(), length(r)).-1f0

    w = window(length(r))
    signal = sum([
        sine_wave(harmonic*f0, 0.5f0/harmonic + 0.5f0rand(Random.default_rng(), Float32), ϕ).(r)
        for harmonic ∈ 1:harmonics
    ])

    signal ./= max(maximum(signal), 1)
    signal .*= A
    signal = (1-β) * w .* signal + β*noise

    input_size = model.stream_filter.cell.n

    xs = Matrix{Float32}(undef, input_size, batches)

    ps, st = Lux.setup(rng, model.stream_filter)
    for i ∈ 1:batches
        batch_range =  1 + (i-1)*step : i*step
        x, st = model.stream_filter(signal[batch_range], ps, st)

        xs[:,i] = x
    end

    xs = xs[:,end-N+1:end]
    ys = repeat([if F0_FLOOR ≤ f0 ≤ F0_CEIL f0 else 0f0 end], size(xs)[end])

    return xs, ys 
end

function create_synthesized_training_data_batch(model, harmonics, N)
    n = model.stream_filter.cell.n
    f0s = Vector{Float32}(0:0.5:MEL_HZ_INTERSECTION)
    m = length(f0s)

    train_filtered_sounds = Array{Float32, 2}(undef, n, m * N * harmonics)
    train_f0s = Array{Float32, 2}(undef, 1, m * N * harmonics)

    for (batch, (hs, f0)) ∈ enumerate(Iterators.product(1:harmonics, f0s))
        s, f = create_synthesized_training_data(
            model, f0, 0.1f0 .+ 0.9f0*rand(Float32), if f0 < 1 0f0 else 10*rand(Float32) end;
            harmonics=hs,
            β=rand(Float32)/10,
            N=N
        )

        batch_range = 1+(batch-1)*N:batch*N
        train_filtered_sounds[:,batch_range] = s
        train_f0s[:, batch_range] = f
    end

    return train_filtered_sounds, normalize_pitch.(train_f0s)
end

function concat_training_data(xs, ys, fade_duration=size(xs)[1]÷3)
    xs = copy(xs)
    ys = copy(ys)

    w = DSP.Windows.hanning(2*fade_duration)
    fade_in_range = 1:fade_duration
    fade_out_range = size(xs)[1]-fade_duration+1:size(xs)[1]

    fade_in = w[1:fade_duration]
    fade_out = w[fade_duration+1:end]

    xs[fade_in_range,:] .*= fade_in
    xs[fade_out_range,:] .*= fade_out

    m = size(xs)[2]
    full_range = shuffle(1:m)
    l_range = full_range[1:end÷2]
    r_range = full_range[end÷2+1:end]

    xs = vcat(xs[:,l_range], xs[:,r_range])
    ys = vcat(ys[:,:,l_range], ys[:,:,r_range])

    return xs, ys
end

function train_pitch_detector(
    model::NeuralPitchDetector, ps, st::NamedTuple, training_data;
    batchsize = 1, epochs = 1, epoch_cb = nothing, batch_cb = nothing, optimiser = Optimisers.Adam(), distributed_backend = nothing
)
    data = training_data[1:2], training_data[end]
    loader = DataLoader(training_data; batchsize=batchsize, shuffle=true)

    function loss_function(model, ps, st, data)
        samples, target_f0s = data
        pred_f0s, st_estimator = model.estimator(samples, ps.estimator, st.estimator)
        loss = sum(abs2, denormalize_pitch.(pred_f0s) - denormalize_pitch.(target_f0s))
        return loss, st, (nothing,)
    end

    return train(
        model, loader, loss_function;
        ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb, batch_cb = batch_cb, distributed_backend = distributed_backend
    )
end
