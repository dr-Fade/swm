include("../nn/training_utils.jl")
include("../sound_utils/neural_pitch_detector.jl")

using ComponentArrays

sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

function get_training_data(model::NeuralPitchDetector, f0::Float32, A::Float32, ϕ::Float32; harmonics=1, β::Float32=0f0, N=nothing, window=ones32)
    N = isnothing(N) ? floor(model.sample_rate / f0) |> Int : N

    range = (0:2*Int(model.sample_rate * N / PITCH_DETECTION_SAMPLE_RATE)) ./ model.sample_rate
    ϕ = f0 == 0f0 ? 0f0 : ϕ
    noise = 2*rand32(Random.default_rng(), length(range)).-1f0

    res = sum([
        sine_wave(harmonic*f0, if harmonic == 1 1f0 else 2*rand(Float32)/harmonic end, ϕ).(range)
        for harmonic ∈ 1:harmonics
    ])
    res ./= max(maximum(res), 1)
    res .*= A
    reset!(model.fir_filter)
    sound = DSP.filt(model.fir_filter, (1-β)*res + β*noise)[end-N+1:end]

    return [sound .* (window(length(sound)) |> Vector{Float32});;]
end

function create_synthesized_training_data(model::NeuralPitchDetector, f0s::Vector{Float32}, As::Vector{Float32}, ϕs::Vector{Float32}; harmonics::Int=1, β=0f0, sample_rate=PITCH_DETECTION_SAMPLE_RATE, N=nothing, window=ones32)
    N = isnothing(N) ? floor(sample_rate / minimum(f0s)) |> Int : N

    input_frames = Vector{Matrix{Float32}}()
    target_f0s = Vector{Vector{Float32}}()
    for (f0, A, ϕ) ∈ zip(f0s, As, ϕs)
        frames = get_training_data(model, f0, A, ϕ; harmonics=harmonics, β=β, N=N, window=window)
        push!(input_frames, frames)
        push!(target_f0s, repeat([f0], size(frames)[1]))
    end

    input_frames = hcat(input_frames...)
    target_f0s = hcat(target_f0s...)

    return input_frames, target_f0s
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
    ys = vcat(ys[:,l_range], ys[:,r_range])

    return xs, ys
end

function train_pitch_detector(
    model::NeuralPitchDetector, ps, st::NamedTuple, training_data;
    batchsize = 1, epochs = 1, epoch_cb = nothing, batch_cb = nothing, optimiser = Optimisers.Adam(), distributed_backend = nothing
)

    loader = DataLoader(training_data; batchsize=batchsize, shuffle=true)

    function loss_function(model, ps, st, data)
        xs, ys = data

        loss = 0f0

        N, batch_n = size(xs)
        sequence = reshape(xs[1:end-N%model.decimated_input_n,:], model.decimated_input_n, N÷model.decimated_input_n, batch_n)
        for (memory_value, y) ∈ zip(Recurrence(model.memory; return_sequence=true)(sequence, ps.memory, st.memory)[1], eachrow(ys))
            pred_y = model.estimator(memory_value, ps.estimator, st.estimator)[1] .|> denormalize_pitch
            loss += sum(abs2, y' - pred_y)
        end

        regularization = sum(abs, ComponentArray(ps)) / length(ComponentArray(ps))
        return regularization + loss / batchsize, st, (nothing,)
    end

    return train(
        model, loader, loss_function;
        ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb, batch_cb = batch_cb, distributed_backend = distributed_backend
    )
end

function pitch_detector_training_routine(model, ps, st)
    train_filtered_sounds = Vector{Matrix{Float32}}()
    train_f0s = Vector{Matrix{Float32}}()

    for i in shuffle(1:10)
        s, f = create_synthesized_training_data(
            model,
            vcat(zeros32(zeros_n), collect(range(1, F0_CEIL, n))),
            0.1f0 .+ 0.7f0*rand32(Random.default_rng(), n+zeros_n),
            10*rand32(Random.default_rng(), n+zeros_n);
            harmonics=i,
            β=0.01f0,
            N=256
        )

        push!(train_filtered_sounds, s)
        push!(train_f0s, f)
    end
    train_filtered_sounds = hcat(train_filtered_sounds...)
    train_f0s = hcat(train_f0s...)

    data =  (train_filtered_sounds, train_f0s)

    return train_pitch_detector(model, ps, st, data; batchsize = 128, epochs = 1, optimiser = Optimisers.Adam(0.0001))
end