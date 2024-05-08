using BSON: @load, @save
using DynamicalSystems, DifferentialEquations, ComponentArrays, DiffEqFlux

include("../nn/unitary_block.jl")
include("../nn/activation_functions.jl")
include("../sound_utils/dio.jl")
include("../sound_utils/mel.jl")

const FEATURE_EXTRACTION_SAMPLE_RATE::Int32 = 16000
const LATENT_DIMS = 10

const F0_CEIL::Float32 = 600f0
const F0_FLOOR::Float32 = 60f0

# the ode trajectories may evolve much slower than the actual time domain signal
# we need to scale the integration step so that it's larger than the Δt between signal samples
const LATENT_SPACE_SAMPLE_RATE_SCALER::Float32 = 1000


# the scanner is a recurrent network because DIO requires f0 info from the previous frame
struct FeatureScanner <: Lux.AbstractRecurrentCell{false, false}
    dio::DIO
    mfcc_filter_bank::Matrix{Float32}
    input_n::Int
    output_n::Int

    function FeatureScanner()
        dio = DIO(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE; noise_floor = 0.01)
        fftn = nextfastfft(dio.frame_length)
        fftfreq = rfftfreq(fftn, FEATURE_EXTRACTION_SAMPLE_RATE)
        mfcc_filter_bank = get_mel_filter_banks(Vector{Float32}(fftfreq))
        return new(dio, mfcc_filter_bank, dio.frame_length, size(mfcc_filter_bank)[1]+2)
    end
end

function (fs::FeatureScanner)(xs::AbstractMatrix, ps, st::NamedTuple)
    return fs((xs, (repeat([(value=0f0, confidence=0f0, loudness=0f0)], size(xs)[2]),)), ps, st)
end

@load "$(@__DIR__)/feature_means.bson" feature_means
@load "$(@__DIR__)/feature_stds.bson" feature_stds

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))

function (fs::FeatureScanner)((xs, (previous_f0s,))::Tuple{<:AbstractMatrix, Tuple{Vector{F0Estimate}}}, ps, st::NamedTuple)
    ys, f0s = [
        begin
            f0 = fs.dio(Vector(x); previous_estimate = f0)
            spectrogram = DSP.periodogram(x; fs = fs.dio.sample_rate, window=blackman)
            mc = mfcc(spectrogram; filter_bank = fs.mfcc_filter_bank)

            (([f0...; mc]) .- feature_means) ./ feature_stds, f0
        end
        for (x, f0) ∈ zip(eachcol(xs), previous_f0s)
    ] |> unzip

    return (reduce(hcat, ys), (f0s,)), st
end

struct hNODEVocoder <: Lux.AbstractExplicitContainerLayer{(:feature_scanner, :control, :encoder, :decoder)}
    feature_scanner::Lux.AbstractExplicitLayer
    control::Lux.AbstractExplicitLayer
    encoder::Lux.AbstractExplicitLayer
    decoder::Lux.AbstractExplicitLayer
    ode::Lux.AbstractExplicitLayer
    ode_axes::Tuple

    resampling_rate::Float32
    resample_filter::Vector{Float64}
    sample_rate::Integer
    input_n::Integer
    output_n::Integer
    Δt::Float32
    T::Float32

    function hNODEVocoder(
        sample_rate;
        output_n = Integer(sample_rate ÷ 1000),
        # whether or not to include the speaker model to go from eigenvoice to specific speaker
        speaker = true
    )
        # analyzers to get MFCC, F0, aperiodicity, etc
        feature_scanner = FeatureScanner()

        # input should contain enough samples to satisfy the analyzers' requirements
        # output has the same length as input
        resampled_input_n = feature_scanner.input_n

        # models to convert into and from the latent space
        encoder_input = Int(FEATURE_EXTRACTION_SAMPLE_RATE÷F0_FLOOR)
        conv_kernel_size = Integer(FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_CEIL)
        conv_output_dims = encoder_input - conv_kernel_size + 1
        encoder = Lux.Chain(
            x -> reshape(x[1:encoder_input,:], encoder_input, 1, size(x)[2]),
            Lux.Conv((conv_kernel_size,), 1=>1, sqrt_activation),
            x -> x[:,1,:],
            Lux.Dense(conv_output_dims, LATENT_DIMS)
        )
        decoder = Lux.Chain(
            Lux.Dense(LATENT_DIMS, LATENT_DIMS, sqrt_activation),
            Lux.Dense(LATENT_DIMS, 1)
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
        T = output_n * Δt

        base_control = Lux.Chain(
            Lux.Dense(feature_scanner.output_n, params_n, sqrt_activation),
            Lux.Dense(params_n, params_n)
        )

        # derive the required modulation to the base control from the input features to arrive at the wanted speaker
        speaker_control = if speaker
            speaker_chain = Lux.Chain(
                Lux.Dense(params_n, params_n, sqrt_activation),
                Lux.Dense(params_n, params_n)
            )
            Lux.SkipConnection(speaker_chain, +)
        else
            Lux.WrappedFunction(identity)
        end

        control = Lux.Chain(;
            base_control = base_control,
            speaker_control = speaker_control
        )

        return new(
            feature_scanner |> Lux.StatefulRecurrentCell,
            control,
            encoder,
            decoder,
            ode,
            ode_axes,
            FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate,
            resample_filter(FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate),
            sample_rate,
            resample(Lux.zeros32(resampled_input_n), sample_rate / FEATURE_EXTRACTION_SAMPLE_RATE) |> length,
            output_n,
            Δt,
            T
        )
    end
end

Lux.initialstates(rng::AbstractRNG, m::hNODEVocoder) = (
    # use the passed in rng for noise generation
    rng = rng,
    # states for all nested models in case they are recurrent
    feature_scanner = Lux.initialstates(rng, m.feature_scanner),
    control = Lux.initialstates(rng, m.control),
    encoder = Lux.initialstates(rng, m.encoder),
    decoder = Lux.initialstates(rng, m.decoder),
    ode = Lux.initialstates(rng, m.ode),
    # a flag to track whether or not the input has been resampled
    resampled = false
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
resample_if_needed(m::hNODEVocoder, sound::AbstractMatrix, st::NamedTuple) =
    if !st.resampled && FEATURE_EXTRACTION_SAMPLE_RATE != m.sample_rate
        resample(sound, m.resampling_rate, m.resample_filter; dims = 1)
    else
        sound
    end

# first stage - embed the sound into the latent space
function (m::hNODEVocoder)(sound::AbstractMatrix, ps, st)
    sound = resample_if_needed(m, sound, st)
    u0, st_encoder = m.encoder(sound, ps.encoder, st.encoder)
    return m((sound, (u0,)), ps, (st..., encoder = st_encoder, resampled = true))
end

# second stage - extract the feature vector
function (m::hNODEVocoder)(
    (sound, (u0,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
    ps,
    st::NamedTuple
)
    sound = resample_if_needed(m, sound, st)
    features, st_feature_scanner = m.feature_scanner(sound, ps.feature_scanner, st.feature_scanner)

    return m((sound, (u0, features)), ps, (st..., feature_scanner = st_feature_scanner, resampled = true))
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

    return (ys, (un,)), (st..., control = control_st, resampled = false)
end
