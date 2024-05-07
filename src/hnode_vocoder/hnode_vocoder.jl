using BSON: @load, @save
using DynamicalSystems

include("../nn/unitary_block.jl")
include("../sound_utils/frames.jl")

const FEATURE_EXTRACTION_SAMPLE_RATE::Int32 = 16000
const LATENT_DIMS = 10

# the scanner is a recurrent network because DIO requires f0 info from the previous frame
struct FeatureScanner <: Lux.AbstractRecurrentCell{false, false}
    dio::DIO
end

const F0_CEIL::Float32 = 600f0
const F0_FLOOR::Float32 = 60f0
const ENCODER_INPUT_N::Integer = Integer(FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_FLOOR ÷ 2)

function (fs::FeatureScanner)(xs::AbstractMatrix, ps, st)
    return fs((xs, (repeat([(value=0f0, confidence=0f0, loudness=0f0)], size(xs)[2]),)), ps, st)
end

@load "$(@__DIR__)/feature_means.bson" feature_means
@load "$(@__DIR__)/feature_stds.bson" feature_stds

function (fs::FeatureScanner)((xs, (previous_f0s,))::Tuple{<:AbstractMatrix, Tuple{Vector{F0Estimate}}}, ps, st::NamedTuple)
    ys, f0s = [
        begin
            loudness, f0, mc, coded_aperiodicity, _ = get_features(x[:,1], fs.dio, f0)
            (([loudness; f0[1]; mc; coded_aperiodicity][:,1] |> Vector{Float32}) .- feature_means) ./ feature_stds, f0, loudness
        end
        for (x, f0) ∈ zip(eachcol(xs), previous_f0s)
    ] |> unzip

    return (reduce(hcat, ys), (f0s,)), st
end


struct RollingFilter <: Lux.AbstractRecurrentCell{false, false}
    filter::Lux.AbstractExplicitLayer
    parameter_generator::Lux.AbstractExplicitLayer
    filter_axes::Tuple
    Δt::Float32

    function RollingFilter(n; Δt=1.0f0)
        filter = Lux.SkipConnection(Lux.Dense(n, n), +)
        filter_axes = Lux.initialparameters(Random.default_rng(), filter) |> ComponentArray |> getaxes
        filter_params_n = Lux.parameterlength(filter)

        parameter_generator = Lux.Chain(
            Lux.Dense(n, filter_params_n, sqrt_activation),
            Lux.Dense(filter_params_n, filter_params_n)
        )

        return new(filter, parameter_generator, filter_axes, Δt)
    end
end

Lux.initialparameters(rng::AbstractRNG, m::RollingFilter) = (
    parameter_generator = Lux.initialparameters(rng, m.parameter_generator),
)

Lux.initialstates(rng::AbstractRNG, m::RollingFilter) = (
    rng = Lux.replicate(rng),
    filter = Lux.initialstates(rng, m.filter),
    parameter_generator = Lux.initialstates(rng, m.parameter_generator)
)

function (m::RollingFilter)(x::AbstractMatrix, ps, st)
    return (x, (x,)), ps, st
end

function (m::RollingFilter)((x, (carry,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}}, ps, st)
    filter_params, generator_st = m.parameter_generator(m.Δt.*(x .- carry), ps.parameter_generator, st.parameter_generator)
    filtered_result, filter_st = m.filter(x, ComponentArray(vec(filter_params), m.filter_axes), st.filter)
    return (filtered_result, (filtered_result,)), (st..., filter = filter_st, parameter_generator = generator_st)
end

# the ode trajectories may evolve much slower than the actual time domain signal
# we need to scale the integration step so that it's larger than the Δt between signal samples
const LATENT_SPACE_SAMPLE_RATE_SCALER::Float32 = 1000

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
        dio = DIO(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE; noise_floor = 0.01)
        feature_scanner = FeatureScanner(dio) |> Lux.StatefulRecurrentCell

        # input should contain enough samples to satisfy the analyzers' requirements
        # output has the same length as input
        resampled_input_n = dio.frame_length
        N = resample(Lux.zeros32(resampled_input_n), sample_rate / FEATURE_EXTRACTION_SAMPLE_RATE) |> length

        # models to convert into and from the latent space
        encoder_input = Int(2*FEATURE_EXTRACTION_SAMPLE_RATE÷F0_FLOOR)
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
        features_n = get_flat_features(Lux.zeros32(resampled_input_n), dio) |> length

        Δt = LATENT_SPACE_SAMPLE_RATE_SCALER / sample_rate
        T = output_n * Δt

        base_control = Lux.Chain(
            Lux.Dense(features_n, params_n, sqrt_activation),
            Lux.Dense(params_n, params_n)
        )

        # derive the required modulation to the base control from the input features to arrive at the wanted speaker
        speaker_control = if speaker
            Lux.Chain(
                Lux.Dense(features_n, params_n, sqrt_activation),
                Lux.Dense(params_n, params_n)
            )
        else
            Lux.WrappedFunction(x ->
                if isa(x, Vector)
                    zeros(Float32, params_n)
                else
                    zeros(Float32, params_n, size(x)[2])
                end
            )
        end

        control = Lux.Parallel(+;
            base_control = base_control,
            speaker_control = speaker_control
        )

        return new(
            feature_scanner,
            control,
            encoder,
            decoder,
            ode,
            ode_axes,
            FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate,
            resample_filter(FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate),
            sample_rate,
            N,
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
