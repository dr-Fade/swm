using Lux, Random, DSP, FFTW, Statistics

include("stream_filter.jl")
include("mel.jl")

PITCH_DETECTION_SAMPLE_RATE::Int = 4000

const F0_CEIL::Float32 = 600f0
const F0_FLOOR::Float32 = 60f0

normalize_pitch(f0) = hz_to_mel(f0) / MEL_HZ_INTERSECTION
denormalize_pitch(normalized_f0) = mel_to_hz(normalized_f0 * MEL_HZ_INTERSECTION)

conv_1d_output_dims(n, k; depth=1) =
    if depth == 1
        n - k + 1
    else
        conv_1d_output_dims(n, k; depth=depth-1) - k + 1
    end

conv_1d(n, k; depth=1) = Lux.Chain(
    Lux.ReshapeLayer((n, 1)),
    [Lux.Conv((k,), 1=>1) for _ in 1:depth],
    Lux.FlattenLayer(),
), conv_1d_output_dims(n,k;depth=depth)

struct NeuralPitchDetector <: Lux.AbstractExplicitLayer
    time_domain_n::Int
    padded_sound_n::Int
    spectrogram_n::Int
    cepstrum_n::Int
    min_quefrency_i::Int
    max_quefrency_i::Int

    hidden_output_n::Int

    window::Vector{Float32}

    stream_filter::Lux.StatefulRecurrentCell
    time_domain_estimator::Lux.AbstractExplicitLayer
    frequency_domain_estimator::Lux.AbstractExplicitLayer
    quefrency_domain_estimator::Lux.AbstractExplicitLayer

    sample_rate::Float32

    function NeuralPitchDetector(sample_rate::Int)
        time_domain_n = 3*PITCH_DETECTION_SAMPLE_RATE ÷ F0_FLOOR |> Int
        padded_sound_n = PITCH_DETECTION_SAMPLE_RATE |> nextfastfft
        frequency_domain_n = (zeros(Float32, padded_sound_n) |> rfft |> length)

        quefrencies = rfftfreq(frequency_domain_n, 1)
        min_quefrency_i = findfirst(q -> q ≥ 1/F0_CEIL, quefrencies)
        max_quefrency_i = length(quefrencies)

        quefrency_domain_n = max_quefrency_i - min_quefrency_i + 1
        hidden_output_n = 32

        filter_window_width = 2*(sample_rate / F0_CEIL) |> floor |> Int
        filter_window = blackman(filter_window_width) |> Vector{Float32} |> FIRWindow
        h = digitalfilter(DSP.Lowpass(2*F0_CEIL; fs=sample_rate), filter_window) |> Vector{Float32}
        stream_filter = StreamFilter(time_domain_n, FIRFilter(h, PITCH_DETECTION_SAMPLE_RATE // sample_rate))

        spectrogram_window = blackman(time_domain_n) |> Vector{Float32}

        time_domain_estimator = Lux.Chain(
            Lux.ReshapeLayer((1, time_domain_n)),
            Lux.RNNCell(1 => hidden_output_n) |> Recurrence,
            Lux.Dense(hidden_output_n, hidden_output_n, tanh)
        )

        conv, conv_dims = conv_1d(frequency_domain_n, Int(F0_CEIL); depth=2)
        frequency_domain_estimator = Lux.Chain(
            conv,
            Lux.Dense(conv_dims, conv_dims, tanh),
            Lux.Dense(conv_dims, hidden_output_n, tanh)
        )

        quefrency_domain_estimator = Lux.Chain(
            Lux.Dense(quefrency_domain_n, quefrency_domain_n, tanh),
            Lux.Dense(quefrency_domain_n, hidden_output_n, tanh)
        )

        return new(
            time_domain_n, padded_sound_n, frequency_domain_n, quefrency_domain_n, min_quefrency_i, max_quefrency_i, hidden_output_n,
            spectrogram_window,
            stream_filter |> Lux.StatefulRecurrentCell,
            time_domain_estimator, frequency_domain_estimator, quefrency_domain_estimator,
            sample_rate
        )
    end

end

Lux.initialparameters(rng::AbstractRNG, npd::NeuralPitchDetector) = (
    Wt = rand32(rng, 1, npd.hidden_output_n),
    Wf = rand32(rng, 1, npd.hidden_output_n),
    Wq = rand32(rng, 1, npd.hidden_output_n),
    b = rand32(rng, 1),
    stream_filter=Lux.initialparameters(rng, npd.time_domain_estimator),
    time_domain_estimator=Lux.initialparameters(rng, npd.time_domain_estimator),
    frequency_domain_estimator=Lux.initialparameters(rng, npd.frequency_domain_estimator),
    quefrency_domain_estimator=Lux.initialparameters(rng, npd.quefrency_domain_estimator)
)

Lux.initialstates(rng::AbstractRNG, npd::NeuralPitchDetector) = (
    stream_filter=Lux.initialstates(rng, npd.time_domain_estimator),
    time_domain_estimator=Lux.initialstates(rng, npd.time_domain_estimator),
    frequency_domain_estimator=Lux.initialstates(rng, npd.frequency_domain_estimator),
    quefrency_domain_estimator=Lux.initialstates(rng, npd.quefrency_domain_estimator)
)

function (npd::NeuralPitchDetector)(sound::AbstractMatrix, ps, st)
    filtered_sound, stream_filter_st = npd.stream_filter(sound, ps.stream_filter, st.stream_filter)
    padded_sound = vcat(filtered_sound .* npd.window, zeros(Float32, npd.padded_sound_n - npd.time_domain_n, size(filtered_sound)[2]))
    spectrogram = (
        s = log10.(abs.(rfft(padded_sound .- mean.(eachcol(padded_sound))', 1)) .+ eps(Float32));
        s .- mean.(eachcol(s))'
    )
    cepstrum = (
        c = log10.(abs.(rfft(spectrogram, 1)[model.min_quefrency_i:model.max_quefrency_i,:]) .+ eps());
        c .- mean.(eachcol(c))'
    )
    return npd((filtered_sound, spectrogram, cepstrum), ps, (st..., stream_filter = stream_filter_st))
end

function (npd::NeuralPitchDetector)((sound, spectrogram, cepstrum), ps, st)
    t, _ = npd.time_domain_estimator(sound, ps.time_domain_estimator, st.time_domain_estimator)
    f, _ = npd.frequency_domain_estimator(spectrogram, ps.frequency_domain_estimator, st.frequency_domain_estimator)
    q, _ = npd.quefrency_domain_estimator(cepstrum, ps.quefrency_domain_estimator, st.quefrency_domain_estimator)

    f0 = σ(ps.Wt*t + ps.Wf*f + ps.Wq*q .+ ps.b) .|> denormalize_pitch

    return f0, st
end
