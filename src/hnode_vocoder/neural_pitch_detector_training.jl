include("../nn/training_utils.jl")
include("../sound_utils/neural_pitch_detector.jl")

using ComponentArrays

sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

function get_training_data(model::NeuralPitchDetector, f0::Float32, A::Float32, ϕ::Float32; harmonics=1, β::Float32=0f0, N=nothing, window=ones32)
    N = isnothing(N) ? floor(model.sample_rate / f0) |> Int : N

    range = (0:N-1) ./ model.sample_rate
    ϕ = f0 == 0f0 ? 0f0 : ϕ
    noise = 2*rand32(Random.default_rng(), N).-1f0

    res = sum([
        sine_wave(harmonic*f0, if harmonic == 1 1f0 else 2*rand(Float32)/harmonic end, ϕ).(range)
        for harmonic ∈ 1:harmonics
    ])
    res ./= max(maximum(res), 1)
    res .*= A
    sound = (1-β)*res + β*noise

    return [sound .* (window(length(sound)) |> Vector{Float32});;]
end

function create_synthesized_training_data(model::NeuralPitchDetector, f0s::Vector{Float32}, As::Vector{Float32}, ϕs::Vector{Float32}; harmonics::Int=1, β=0f0, sample_rate=PITCH_DETECTION_SAMPLE_RATE, N=nothing, window=ones32)
    N = isnothing(N) ? floor(sample_rate / minimum(f0s)) |> Int : N

    input_frames = Vector{Matrix{Float32}}()
    target_f0s = Vector{Float32}()
    for (f0, A, ϕ) ∈ zip(f0s, As, ϕs)
        frames = get_training_data(model, f0, A, ϕ; harmonics=harmonics, β=β, N=N, window=window)
        push!(input_frames, frames)
        push!(target_f0s, f0)
    end

    input_frames = hcat(input_frames...)
    target_f0s = target_f0s'

    return input_frames, target_f0s

    # return (filtered_sounds .- filtered_sound_means) ./ filtered_sound_stds,
    #     (spectrogram .- spectrogram_means) ./ spectrogram_stds,
    #     (cepstrum .- cepstrum_means) ./ cepstrum_stds,
    #     target_f0s
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

function estimate_f0(model, ps, st, signal)
    pred_ys = zeros32(1, size(signal)[2])
    for s ∈ eachslice(reshape(signal, 1, size(signal)...); dims=2)
        pred_ys, st = model(s, ps, st)
    end
    return denormalize_pitch.(pred_ys)
end

function train_pitch_detector(
    model::Lux.AbstractExplicitLayer, ps, st::NamedTuple, training_data;
    batchsize = 1, epochs = 1, epoch_cb = nothing, batch_cb = nothing, optimiser = Optimisers.Adam(), distributed_backend = nothing
)

    loader = DataLoader(training_data; batchsize=batchsize, shuffle=true)

    function loss_function(model, ps, st, data)
        xs, ys = data

        loss = 0f0

        for memory_value ∈ Recurrence(model.memory; return_sequence=true)(reshape(xs, 1, size(xs)...), ps.memory, st.memory)[1]
            pred_ys = model.estimator(memory_value, ps.estimator, st.estimator)[1] .|> denormalize_pitch
            loss += sum(abs2, ys .- pred_ys)
        end

        regularization = sum(abs, ComponentArray(ps)) / length(ComponentArray(ps))
        return regularization + loss / batchsize, st, (nothing,)
    end

    return train(
        model, loader, loss_function;
        ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb, batch_cb = batch_cb, distributed_backend = distributed_backend
    )
end
