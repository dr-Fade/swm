using Lux, Random, DSP, FFTW, Statistics

include("stream_filter.jl")
include("mel.jl")
include("../nn/activation_functions.jl")

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

struct NeuralPitchDetector <: Lux.AbstractExplicitContainerLayer{(:memory, :estimator)}
    fir_filter::FIRFilter
    memory::Lux.AbstractRecurrentCell
    estimator::Lux.AbstractExplicitLayer
    sample_rate::Float32

    function NeuralPitchDetector(sample_rate::Int)
        filter_window_width = 2*(sample_rate / F0_CEIL) |> floor |> Int
        filter_window = blackman(filter_window_width) |> Vector{Float32} |> FIRWindow
        h = digitalfilter(DSP.Lowpass(2*F0_CEIL; fs=sample_rate), filter_window) |> Vector{Float32}
        fir_filter = FIRFilter(h, PITCH_DETECTION_SAMPLE_RATE // sample_rate)

        memory_n = 256
        memory = Lux.RNNCell(1 => memory_n; init_state=rand32)
        estimator = Lux.Chain(
            Lux.Dense(memory_n, memory_n, tanh),
            Lux.Dense(memory_n, 1, σ)
        )

        return new(
            fir_filter,
            memory,
            estimator,
            sample_rate
        )
    end

end

Lux.initialparameters(rng::AbstractRNG, npd::NeuralPitchDetector) = (
    memory=Lux.initialparameters(rng, npd.memory),
    estimator=Lux.initialparameters(rng, npd.estimator),
)

Lux.initialstates(rng::AbstractRNG, npd::NeuralPitchDetector) = (
    memory=Lux.initialstates(rng, npd.memory),
    estimator=Lux.initialstates(rng, npd.estimator),
)

function (npd::NeuralPitchDetector)(samples, ps, st)
    reset!(npd.fir_filter)
    return npd((samples, zeros32(1, size(samples)[end])), ps, st)
end

function (npd::NeuralPitchDetector)((samples, prev_f0)::Tuple, ps, st)
    decimated_samples = npd.sample_rate == PITCH_DETECTION_SAMPLE_RATE ? samples : DSP.filt(npd.fir_filter, samples)
    return if length(decimated_samples) > 0
        rnn_sequence = reshape(decimated_samples, 1, size(decimated_samples)...)
        memory_value, memory_st = Recurrence(npd.memory)(rnn_sequence, ps.memory, st.memory)
        f0s, estimator_st = npd.estimator(memory_value, ps.estimator, st.estimator)
        denormalize_pitch.(f0s), (st..., memory = memory_st, estimator = estimator_st)
    else
        prev_f0
    end
end
