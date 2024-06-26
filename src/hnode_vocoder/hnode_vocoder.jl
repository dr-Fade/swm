using BSON: @load, @save
using DynamicalSystems, DifferentialEquations, ComponentArrays, DiffEqFlux, Random

include("../nn/time_step_aware_rnn.jl")
include("../nn/activation_functions.jl")
include("../nn/conv_1d.jl")
include("../sound_utils/dio.jl")
include("../sound_utils/cheaptrick.jl")
include("../sound_utils/mel.jl")
include("../sound_utils/stream_filter.jl")

const FEATURE_EXTRACTION_SAMPLE_RATE::Int = 16000
const LATENT_DIMS = 10

const F0_CEIL::Float32 = 600f0
const F0_FLOOR::Float32 = 60f0

const F0_CEIL_MEL::Float32 = hz_to_mel(F0_CEIL)

normalize_pitch(f0) = hz_to_mel(f0) / F0_CEIL_MEL
denormalize_pitch(normalized_f0) = mel_to_hz(normalized_f0 * F0_CEIL_MEL)

# the ode trajectories may evolve much slower than the actual time domain signal
# we need to scale the integration step so that it's larger than the Δt between signal samples
const LATENT_SPACE_SAMPLE_RATE_SCALER::Float32 = 1000

const MFCC_SIZE::Int = 29

# the scanner is a recurrent network because DIO requires f0 info from the previous frame
struct FeatureScanner <: Lux.AbstractRecurrentCell{false, false}
    dio::DIO
    cheaptrick::CheapTrick
    mfcc_filter_bank::Matrix{Float32}
    input_n::Int
    hop_size::Int
    output_n::Int

    function FeatureScanner(hop_size::Int)
        dio = DIO(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE; noise_floor = 0.02)
        cheaptrick = CheapTrick(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE)

        input_n = max(
            dio.decimated_frame_length / DOWNSAMPLED_RATE * FEATURE_EXTRACTION_SAMPLE_RATE,
            cheaptrick.max_frame_size
        ) |> round |> Int

        fftfreq = rfftfreq(input_n, FEATURE_EXTRACTION_SAMPLE_RATE)
        mfcc_filter_bank = get_mel_filter_banks(Vector{Float32}(fftfreq); k=MFCC_SIZE+1)
        output_n = size(mfcc_filter_bank)[1]+2

        return new(dio, cheaptrick, mfcc_filter_bank, input_n, hop_size, output_n)
    end
end

function (fs::FeatureScanner)(xs::AbstractMatrix, ps, st::NamedTuple)
    return fs((xs, (repeat([(value=0f0, confidence=0f0)], size(xs)[2]),)), ps, st)
end

@load "$(@__DIR__)/feature_means.bson" feature_means
@load "$(@__DIR__)/feature_stds.bson" feature_stds

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function (fs::FeatureScanner)((xs, (previous_f0s,))::Tuple{<:AbstractMatrix, Tuple{Vector{F0Estimate}}}, ps, st::NamedTuple)
    ys, f0s = [
        begin
            f0 = fs.dio(Vector(x[end-fs.hop_size+1:end]); previous_estimate = f0)
            spectrogram = fs.cheaptrick(f0.value, Vector(x))
            mc = σ.(mfcc(spectrogram; filter_bank = fs.mfcc_filter_bank))

            [normalize_pitch(f0.value); f0.confidence; rms(x); mc], f0
        end
        for (x, f0) ∈ zip(eachcol(xs), previous_f0s)
    ] |> unzip

    return (reduce(hcat, ys), (f0s,)), st
end

struct hNODEVocoder <: Lux.AbstractExplicitContainerLayer{(:stream_filter, :feature_scanner, :control, :encoder, :decoder)}
    stream_filter::Lux.AbstractExplicitLayer
    feature_scanner::Lux.AbstractExplicitLayer
    control::Lux.AbstractExplicitLayer
    encoder::Lux.AbstractExplicitLayer
    decoder::Lux.AbstractExplicitLayer
    ode::Lux.AbstractExplicitLayer
    ode_axes::Tuple

    sample_rate::Integer
    n::Integer

    Δt::Float32
    T::Float32

    function hNODEVocoder(
        sample_rate;
        n = Integer(sample_rate ÷ F0_CEIL),
        # whether or not to include the speaker model to go from eigenvoice to specific speaker
        speaker = true
    )
        # analyzers to get MFCC, F0, aperiodicity, etc
        feature_scanner = FeatureScanner(n)

        # input should contain enough samples to satisfy the analyzers' requirements
        # output has the same length as input
        frame_size = feature_scanner.input_n

        if n > frame_size
            error("The hop size ($(n)) cannot be larger than the frame size used by the feature scanner ($(frame_size)).")
        end

        # to make the job of the analyzers easier, we filter out the frequencies outside of the human vocal range
        Nh = Int(2*FEATURE_EXTRACTION_SAMPLE_RATE÷F0_CEIL)
        h = digitalfilter(DSP.Bandpass(F0_FLOOR, 4*F0_CEIL; fs=FEATURE_EXTRACTION_SAMPLE_RATE), FIRWindow(ones(Nh)./(Nh)))
        fir_filter = FIRFilter(h, FEATURE_EXTRACTION_SAMPLE_RATE // sample_rate)
        stream_filter = StreamFilter(frame_size, fir_filter; pregain=2f0, gain=2f0)

        # models to convert into and from the latent space
        encoder_n = Integer(FEATURE_EXTRACTION_SAMPLE_RATE÷F0_FLOOR)
        conv, conv_dims = conv_1d(encoder_n, Integer(FEATURE_EXTRACTION_SAMPLE_RATE÷F0_CEIL))
        encoder = Lux.Chain(
            x -> x[1:encoder_n,:],
            conv,
            Lux.Dense(conv_dims, conv_dims, tanh),
            Lux.Dense(conv_dims, LATENT_DIMS)
        )
        decoder = Lux.Chain(
            Lux.Dense(LATENT_DIMS, 2*LATENT_DIMS, sqrt_activation),
            Lux.Dense(2*LATENT_DIMS, 1)
        )
        # model to get trajectories inside the latent space
        # assumes ther are linear plus non-linear parts to the signal
        ode = Lux.Parallel(+;
            name = "ode",
            linear_block = Lux.Dense(LATENT_DIMS, LATENT_DIMS),
            nonlinear_block = Lux.Dense(LATENT_DIMS, LATENT_DIMS, sqrt_activation)
        )
        ode_axes = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> getaxes
        params_n = Lux.parameterlength(ode)

        # infer the basic parameters from features to generate time-domain signal
        Δt = LATENT_SPACE_SAMPLE_RATE_SCALER / sample_rate
        T = n * Δt

        base_control = Lux.Chain(
            Lux.Dense(params_n, params_n, tanh),
            Lux.Dense(params_n, params_n, tanh),
            Lux.Dense(params_n, params_n)
        )
        speaker_control = speaker ?
            copy(base_control) :
            Lux.WrappedFunction(x -> zeros(eltype(x), params_n, size(x)[2]))

        control = Lux.Chain(;
            feature_filter = TimeStepAwareRNN(feature_scanner.output_n, params_n, identity; dt = T) |> Lux.StatefulRecurrentCell,
            mlp = Lux.Parallel(+; base_control = base_control, speaker_control = speaker_control)
        )

        return new(
            stream_filter |> Lux.StatefulRecurrentCell,
            feature_scanner |> Lux.StatefulRecurrentCell,
            control,
            encoder,
            decoder,
            ode,
            ode_axes,
            sample_rate,
            n,
            Δt,
            T
        )
    end
end

Lux.initialstates(rng::AbstractRNG, m::hNODEVocoder) = (
    # use the passed in rng for noise generation
    rng = rng,
    # states for all nested models in case they are recurrent
    stream_filter = Lux.initialstates(rng, m.stream_filter),
    feature_scanner = Lux.initialstates(rng, m.feature_scanner),
    control = Lux.initialstates(rng, m.control),
    encoder = Lux.initialstates(rng, m.encoder),
    decoder = Lux.initialstates(rng, m.decoder),
    ode = Lux.initialstates(rng, m.ode),
    # a flag to track whether or not the input has been filtered
    filtered = false
)

get_encoder(m::hNODEVocoder, ps, st) = begin
    m.encoder, ps.encoder, st.encoder
end

get_decoder(m::hNODEVocoder, ps, st) = begin
    m.decoder, ps.decoder, st.decoder
end

get_control(m::hNODEVocoder, ps, st) = begin
    m.control, ps.control, st.control
end

# Zygote doesn't like DSP functions so if the resampling is not required, skip it
filter_if_needed(m::hNODEVocoder, sound::AbstractMatrix, st::NamedTuple) =
    if !st.filtered && FEATURE_EXTRACTION_SAMPLE_RATE != m.sample_rate
        m.stream_filter(sound)
    else
        sound
    end

# first stage - embed the sound into the latent space
function (m::hNODEVocoder)(sound::AbstractMatrix, ps, st)
    sound = filter_if_needed(m, sound, st)
    u0, st_encoder = m.encoder(sound, ps.encoder, st.encoder)
    return m((sound, (u0,)), ps, (st..., encoder = st_encoder, filtered = true))
end

# second stage - extract the feature vector
function (m::hNODEVocoder)(
    (sound, (u0,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
    ps,
    st::NamedTuple
)
    sound = filter_if_needed(m, sound, st)
    features, st_feature_scanner = m.feature_scanner(sound, ps.feature_scanner, st.feature_scanner)

    return m((sound, (u0, features)), ps, (st..., feature_scanner = st_feature_scanner, filtered = true))
end

# third stage - use the feature vector to get the control for ode and integrate it using the embedded sound as u0
function (m::hNODEVocoder)(
    (_, (u0, features))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix, <:AbstractMatrix}},
    ps,
    st::NamedTuple
)
    tspan = (0.0f0, m.T)
    node = NeuralODE(m.ode, tspan, Tsit5(), saveat=0f0:m.Δt:m.T, save_on=true, save_start=true, save_end=true, verbose=false)

    control, control_st = m.control(features, ps.control, st.control)

    ys, un = [
        begin
            ode_params = ComponentArray(view(control,:,i), m.ode_axes)
            ode_solution = node(view(u0,:,i), ode_params, st.ode)[1].u

            decoded_trajectory = m.decoder(reduce(hcat, ode_solution[1:end-1]), ps.decoder, st.decoder)[1]
            new_u0 = ode_solution[end]

            decoded_trajectory', new_u0
        end
        for i ∈ 1:size(features)[2]
    ] |> unzip

    ys = hcat(ys...)
    un = hcat(un...)

    return (ys, (un,)), (st..., control = control_st, filtered = false)
end
