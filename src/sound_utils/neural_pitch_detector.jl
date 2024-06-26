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
    input_n::Int
    fir_filter::FIRFilter
    decimated_input_n::Int
    memory::Lux.AbstractRecurrentCell
    estimator::Lux.AbstractExplicitLayer
    sample_rate::Float32

    function NeuralPitchDetector(sample_rate::Int)
        filter_window_width = 2*(sample_rate / F0_CEIL) |> floor |> Int
        filter_window = blackman(filter_window_width) |> Vector{Float32} |> FIRWindow
        h = digitalfilter(DSP.Lowpass(2*F0_CEIL; fs=sample_rate), filter_window) |> Vector{Float32}
        fir_filter = FIRFilter(h, PITCH_DETECTION_SAMPLE_RATE // sample_rate)

        decimated_input_n = Int(PITCH_DETECTION_SAMPLE_RATE ÷ F0_CEIL)
        input_n = (decimated_input_n * sample_rate / PITCH_DETECTION_SAMPLE_RATE) |> floor |> Int

        memory_n = 256
        memory = Lux.GRUCell(decimated_input_n => memory_n; init_state=rand32)
        estimator = Lux.Chain(
            Lux.Dense(memory_n, memory_n, tanh),
            Lux.Dense(memory_n, 1, σ)
        )

        return new(
            input_n,
            fir_filter,
            decimated_input_n,
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
    filtered=false
)

function (npd::NeuralPitchDetector)(samples, ps, st)
    reset!(npd.fir_filter)
    decimated_samples =
        if npd.sample_rate == PITCH_DETECTION_SAMPLE_RATE || st.filtered
            samples
        else
            DSP.filt(npd.fir_filter, samples)
        end
    return npd((decimated_samples, zeros32(1, size(samples)[end])), ps, (st..., filtered=true))
end

function full_sequence_f0(npd::NeuralPitchDetector, ps, st, xs)
    N, batch_n = size(xs)
    sequence = reshape(xs[1:end-N%npd.decimated_input_n,:], npd.decimated_input_n, N÷npd.decimated_input_n, batch_n)

    memory_value, memory_st = Recurrence(npd.memory)(sequence, ps.memory, st.memory)
    f0s, estimator_st = npd.estimator(memory_value, ps.estimator, st.estimator)

    return denormalize_pitch.(f0s), (st..., memory = memory_st, estimator = estimator_st)
end

function f0_contour(npd::NeuralPitchDetector, ps, st, xs)
    N, m = size(xs)

    res = Vector{Vector{Float32}}()
    for i ∈ 1:model.decimated_input_n:N-model.decimated_input_n
        ys, st = full_sequence_f0(npd, ps, st, xs[i:i+model.decimated_input_n-1,:])
        push!(res, ys[1,:])
    end

    return hcat(res...)
end

function (npd::NeuralPitchDetector)((samples, prev_f0)::Tuple, ps, st)
    decimated_samples =
        if npd.sample_rate == PITCH_DETECTION_SAMPLE_RATE || st.filtered
            samples
        else
            DSP.filt(npd.fir_filter, samples)
        end
    
    return if size(decimated_samples)[1] ≥ model.decimated_input_n
        full_sequence_f0(npd, ps, st, decimated_samples)
    else
        prev_f0
    end
end
