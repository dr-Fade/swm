using BSON: @load, @save
using DynamicalSystems, DifferentialEquations, ComponentArrays, DiffEqFlux, Random

include("../nn/merge_block.jl")
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
const LATENT_SPACE_SAMPLE_RATE_SCALER::Float32 = 2000

const MFCC_SIZE::Int = 29

# the scanner is a recurrent network because DIO requires f0 info from the previous frame
struct FeatureScanner <: Lux.AbstractRecurrentCell
    dio::DIO
    mfcc_filter_bank::Matrix{Float32}
    input_n::Int
    hop_size::Int
    output_n::Int

    function FeatureScanner(hop_size::Int)
        dio = DIO(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE; noise_floor = 0.02)

        input_n = dio.decimated_frame_length / DOWNSAMPLED_RATE * FEATURE_EXTRACTION_SAMPLE_RATE |> round |> Int

        fftfreq = rfftfreq(input_n, FEATURE_EXTRACTION_SAMPLE_RATE)
        mfcc_filter_bank = get_mel_filter_banks(Vector{Float32}(fftfreq); k=MFCC_SIZE+1)
        output_n = size(mfcc_filter_bank)[1]+2

        return new(dio, mfcc_filter_bank, input_n, hop_size, output_n)
    end
end

function (fs::FeatureScanner)(xs::AbstractMatrix, ps, st::NamedTuple)
    return fs((xs, (repeat([(value=0f0, confidence=0f0)], size(xs)[2]),)), ps, st)
end

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function (fs::FeatureScanner)((xs, (previous_f0s,))::Tuple{<:AbstractMatrix, Tuple{Vector{F0Estimate}}}, ps, st::NamedTuple)
    ys, f0s = [
        begin
            f0 = fs.dio(Vector(x[end-fs.hop_size+1:end]); previous_estimate = f0)
            spectrogram = log10.(DSP.periodogram(x; fs = fs.dio.sample_rate, window=blackman).power .+ eps(Float32))
            mc = mfcc(spectrogram; filter_bank = fs.mfcc_filter_bank)

            [f0.value; f0.confidence; rms(x); mc], f0
        end
        for (x, f0) ∈ zip(eachcol(xs), previous_f0s)
    ] |> unzip
    ys = reduce(hcat, ys)

    return ((f0s=ys[1:2,:], loudness=ys[3:3, :], mfccs=ys[4:end,:]), (f0s,)), st
end

struct hNODEVocoder <: Lux.AbstractLuxContainerLayer{(:stream_filter, :feature_scanner, :control, :encoder, :decoder)}
    stream_filter::Lux.AbstractLuxLayer
    feature_scanner::Lux.AbstractLuxLayer
    control::Lux.AbstractLuxLayer
    encoder::Lux.AbstractLuxLayer
    decoder::Lux.AbstractLuxLayer
    ode::Lux.AbstractLuxLayer
    ode_axes::Tuple

    sample_rate::Integer
    n::Integer

    Δt::Float32
    T::Float32

    function hNODEVocoder(
        sample_rate;
        n = Integer(sample_rate ÷ F0_CEIL),
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
        stream_filter = StreamFilter(frame_size, sample_rate;
            target_sample_rate = FEATURE_EXTRACTION_SAMPLE_RATE,
            f0_floor = F0_FLOOR / 2f0,
            f0_ceil = FEATURE_EXTRACTION_SAMPLE_RATE / 2f0 - 1
        )

        # models to convert into and from the latent space
        encoder_n = Integer(FEATURE_EXTRACTION_SAMPLE_RATE÷F0_FLOOR)
        encoder_conv, envoder_conv_n = conv_1d(encoder_n, Integer(FEATURE_EXTRACTION_SAMPLE_RATE÷F0_CEIL); depth=2)
        encoder = Lux.Chain(
            x -> view(x, 1:encoder_n, :),
            encoder_conv,
            Lux.Dense(envoder_conv_n, envoder_conv_n, leaky_tanh()),
            Lux.Dense(envoder_conv_n, LATENT_DIMS)
        )
        decoder = Lux.Chain(
            SkipConnection(
                Chain(
                    Dense(LATENT_DIMS, 2*LATENT_DIMS, leaky_tanh()),
                    Dense(2*LATENT_DIMS, LATENT_DIMS, leaky_tanh())
                ), +
            ),
            Dense(LATENT_DIMS, 1)
        )
        # model to get trajectories inside the latent space
        # assumes there are linear plus non-linear parts to the signal
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

        control = Lux.Chain(
            MergeLayer((2, 1, MFCC_SIZE) => 4*params_n, +, leaky_tanh()),
            Lux.Dense(4*params_n, 4*params_n, leaky_tanh()),
            Lux.Dense(4*params_n, params_n)
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

    return m((sound, (u0, Tuple(features))), ps, (st..., feature_scanner = st_feature_scanner, filtered = true))
end

# third stage - use the feature vector to get the control for ode and integrate it using the embedded sound as u0
function (m::hNODEVocoder)(
    (_, (u0, features))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix, Tuple}},
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
        for i ∈ 1:size(u0)[2]
    ] |> unzip

    ys = hcat(ys...)
    un = hcat(un...)

    return (ys, (un,)), (st..., control = control_st, filtered = false)
end
