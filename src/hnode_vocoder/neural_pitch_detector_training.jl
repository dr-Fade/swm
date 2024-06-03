include("../nn/training_utils.jl")
include("../sound_utils/neural_pitch_detector.jl")

using ComponentArrays

sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

function sound_to_training_data(model::NeuralPitchDetector, sound, hop_size)
    stream_filter = model.stream_filter
    stream_filter_ps, stream_filter_st = Lux.setup(rng, stream_filter)

    filtered_sounds = Vector{Vector{Float32}}()
    for i ∈ 1:hop_size:length(sound)-hop_size
        frame = sound[i:i+hop_size-1]
        filtered_sound, stream_filter_st = stream_filter(frame, stream_filter_ps, stream_filter_st)
        if PITCH_DETECTION_SAMPLE_RATE * i / model.sample_rate < model.time_domain_n-1
            continue
        end
        push!(filtered_sounds, filtered_sound)
    end
    filtered_sounds = hcat(filtered_sounds...)

    padded_sound = vcat(filtered_sounds .* model.window, zeros(Float32, model.padded_sound_n - model.time_domain_n, size(filtered_sounds)[2]))
    spectrogram = (
        s = log10.(abs.(rfft(padded_sound .- mean.(eachcol(padded_sound))', 1)) .+ eps(Float32));
        s .- mean.(eachcol(s))'
    )
    cepstrum = (
        c = log10.(abs.(rfft(spectrogram, 1)[model.min_quefrency_i:model.max_quefrency_i,:]) .+ eps());
        c .- mean.(eachcol(c))'
    )
    return filtered_sounds, spectrogram, cepstrum
end

function get_training_data(model::NeuralPitchDetector, f0::Float32, A::Float32, ϕ::Float32; harmonics=1, β::Float32=0f0, N=nothing, hop_size)
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

    window = blackman(length(sound)) |> Vector{Float32}

    return sound_to_training_data(model, sound .* window, hop_size)
end

function create_synthesized_training_data(model::NeuralPitchDetector, f0s::Vector{Float32}, As::Vector{Float32}, ϕs::Vector{Float32}; harmonics::Int=1, β=0f0, sample_rate=PITCH_DETECTION_SAMPLE_RATE, N=nothing, hop_size=nothing)
    N = isnothing(N) ? floor(sample_rate / minimum(f0s)) |> Int : N
    hop_size = isnothing(hop_size) ? Integer(sample_rate ÷ 1000) |> Int : hop_size

    filtered_sounds = Vector{Matrix{Float32}}()
    spectrogram = Vector{Matrix{Float32}}()
    cepstrum = Vector{Matrix{Float32}}()
    target_f0s = Vector{Vector{Float32}}()
    for (f0, A, ϕ) ∈ zip(f0s, As, ϕs)
        f, s, c = get_training_data(model, f0, A, ϕ; harmonics=harmonics, β=β, N=N, hop_size=hop_size)
        push!(filtered_sounds, f)
        push!(spectrogram, s)
        push!(cepstrum, c)
        push!(target_f0s, repeat([f0], size(f)[2]))
    end

    filtered_sounds = hcat(filtered_sounds...)
    spectrogram = hcat(spectrogram...)
    cepstrum = hcat(cepstrum...)
    target_f0s = vcat(target_f0s...)'

    return filtered_sounds, spectrogram, cepstrum, target_f0s

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
        sound, spectrogram, cepstrum, ys = data
        xs = (sound, spectrogram, cepstrum)

        pred_ys, _ = model(xs, ps, st)
        loss = sum(abs2, ys .- pred_ys)

        regularization = sum(abs, ComponentArray(ps)) / length(ComponentArray(ps))
        return regularization + loss / batchsize, st, (nothing,)
    end

    return train(
        model, loader, loss_function;
        ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb, batch_cb = batch_cb, distributed_backend = distributed_backend
    )
end

# function train_frequency_domain_pitch_detector(
#     model::Lux.AbstractExplicitLayer, ps, st::NamedTuple, training_data;
#     batchsize = 1, epochs = 1, epoch_cb = nothing, batch_cb = nothing, optimiser = Optimisers.Adam(), distributed_backend = nothing
# )

#     loader = DataLoader(training_data; batchsize=batchsize, shuffle=true)

#     function loss_function(model, ps, st, data)
#         sound, spectrogram, cepstrum, ys = data
#         loss = 0f0

#         ts, _ = model.time_domain_estimator(sound, ps.time_domain_estimator, st.time_domain_estimator)

#         estimate_loss = sum.(abs2, ys .- pred_ys)
#         loss = individual_losses + final_loss

#         regularization = 0#sum(abs, ComponentArray(ps)) / length(ComponentArray(ps))
#         return regularization + loss / batchsize, st, (nothing,)
#     end

#     return train(
#         model, loader, loss_function;
#         ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb, batch_cb = batch_cb, distributed_backend = distributed_backend
#     )
# end