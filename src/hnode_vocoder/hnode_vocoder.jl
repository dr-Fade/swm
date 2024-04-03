using BSON: @load, @save
using DynamicalSystems

include("../nn/hnode.jl")
include("../nn/unitary_block.jl")
include("../nn/antisymmetric_rnn.jl")
include("../sound_utils/frames.jl")

const FEATURE_EXTRACTION_SAMPLE_RATE::Int32 = 16000
const LATENT_DIMS = 10

# the scanner is a recurrent network because DIO requires f0 info from the previous frame
struct FeatureScanner <: Lux.AbstractRecurrentCell{false, false}
    dio::DIO
    cheaptrick::CheapTrick
end

const F0_CEIL::Float32 = 600f0
const F0_FLOOR::Float32 = 60f0
const ENCODER_INPUT_N::Integer = Integer(2 * FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_CEIL)

f0_scaling = 1 / log2(F0_CEIL)
normalize_f0(f0) = log2.(max(f0, 1)) .* f0_scaling

MFCC_BINS_MEANS = [
    1.8214860450883197
    0.11150358086286806
    0.40296419778594844
   -0.03726557445384133
   -0.024752396914953356
   -0.13345687835636597
   -0.13945961262487958
   -0.08544703122366047
   -0.0496579958864385
   -0.0863300218427441
   -0.04799903214594542
   -0.0793286403268334
   -0.0449841612210477
   -0.0775410522512287
   -0.0502043202189909
   -0.07112892873412492
   -0.05535474973035656
   -0.05727726959186062
   -0.04899441672716835
   -0.051117201178009455
]

MFCC_BINS_STDS = [
    1.073383646595744
    0.6228678792361044
    0.4841264245321103
    0.3859067685108518
    0.3119141498935564
    0.2649908878119277
    0.22844971763687744
    0.1921292417308152
    0.18011519501357331
    0.16704252127305141
    0.15278848880919385
    0.14681234412473812
    0.1287666615508793
    0.12247905230402605
    0.10843719019719679
    0.10395637745268518
    0.09440120621080068
    0.09073532837113985
    0.08494315069415431
    0.07840568631635172
]

normalize_mfcc(mfcc) = σ((mfcc .- MFCC_BINS_MEANS) ./ MFCC_BINS_STDS)

CODED_APERIODICITY_MEAN = -2.0900513068124376
CODED_APERIODICITY_STD_DEV = 3.031379481659005
normalize_aperiodicity(aperiodicity) = σ(-(aperiodicity .- CODED_APERIODICITY_MEAN) ./ CODED_APERIODICITY_STD_DEV)

function (fs::FeatureScanner)(xs::AbstractMatrix, ps, st)
    return fs((xs, repeat([(0.0, 0.0)], size(xs)[2])), ps, st)
end

function (fs::FeatureScanner)((xs, previous_f0s)::Tuple{<:AbstractMatrix, Vector{Tuple{Float64, Float64}}}, ps, st::NamedTuple)
    ys, f0s = [
        begin
            loudness, f0, mc, coded_aperiodicity, _ = get_features(x |> Vector{Float32}, fs.dio, fs.cheaptrick, f0)
            [loudness; normalize_f0(f0[1]); normalize_mfcc(mc); normalize_aperiodicity(coded_aperiodicity)][:,1] |> Vector{Float32}, f0
        end
        for (x, f0) ∈ zip(eachcol(xs), previous_f0s)
    ] |> unzip
    return (reduce(hcat, ys), f0s), st
end

# the ode trajectories may evolve much slower than the actual time domain signal
# we need to scale the integration step so that it's larger than the Δt between signal samples
const LATENT_SPACE_SAMPLE_RATE_SCALER::Float32 = 1000

struct hNODEVocoder <: Lux.AbstractExplicitContainerLayer{(:feature_scanner, :control, :encoder, :identity_decoder, :decoder)}
    feature_scanner::Lux.AbstractExplicitLayer
    control::Lux.AbstractExplicitLayer
    encoder::Lux.AbstractExplicitLayer
    identity_decoder::Lux.AbstractExplicitLayer
    decoder::Lux.AbstractExplicitLayer
    ode::Lux.AbstractExplicitLayer
    ode_axes::Tuple

    resampling_rate::Float32
    resample_filter::Vector{Float64}
    sample_rate::Integer
    input_n::Integer
    output_n::Integer

    function hNODEVocoder(
        sample_rate;
        # whether or not to include the speaker model to go from eigenvoice to specific speaker
        speaker = true,
        # specify the number of output samples, otherwise, the output is the same as the input
        output_n = nothing
    )
        # analyzers to get MFCC, F0, aperiodicity, etc
        dio = DIO(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE)
        cheaptrick = CheapTrick(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE)
        feature_scanner = FeatureScanner(dio, cheaptrick) |> Lux.StatefulRecurrentCell

        # input should contain enough samples to satisfy the analyzers' requirements
        # output has the same length as input
        resampled_input_n = max(cheaptrick.max_frame_size, dio.frame_length)
        N = resample(Lux.zeros32(resampled_input_n), sample_rate / FEATURE_EXTRACTION_SAMPLE_RATE) |> length

        # models to convert into and from the latent space
        encoder = Lux.Dense(N, LATENT_DIMS)
        identity_decoder = Lux.Dense(LATENT_DIMS, N)
        decoder = Lux.Dense(LATENT_DIMS, 1)
        # model to get trajectories inside the latent space
        # assumes ther are linear plus non-linear parts to the signal
        ode = Lux.Parallel(+;
            name = "ode",
            linear_block = Lux.Dense(LATENT_DIMS, LATENT_DIMS),
            antisymmetric_block = Lux.Dense(LATENT_DIMS, LATENT_DIMS, variable_power(0.5f0, 0.5f0, 1f0))
        )
        ode_axes = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> getaxes
        params_n = Lux.parameterlength(ode) + 1 # an extra parameter for the white noise level

        # infer the basic parameters from features to generate time-domain signal
        features_n = get_flat_features(Lux.zeros32(resampled_input_n), dio, cheaptrick) |> length
        base_control = Lux.Chain(
            Lux.Dense(features_n, (features_n + params_n) ÷ 2, tanh),
            Lux.Dense((features_n + params_n) ÷ 2, params_n)
        )

        # use a residual network to arrive at the wanted speaker
        speaker_control = if speaker
            Lux.SkipConnection(Lux.Dense(params_n, params_n, tanh), +)
        else
            Lux.NoOpLayer()
        end
        control = Lux.Chain(;
            base_control = base_control,
            speaker_control = speaker_control
        )

        return new(
            feature_scanner,
            control,
            encoder,
            identity_decoder,
            decoder,
            ode,
            ode_axes,
            FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate,
            resample_filter(FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate),
            sample_rate,
            N,
            isnothing(output_n) ? N : output_n
        )
    end
end

Lux.initialstates(rng::AbstractRNG, m::hNODEVocoder) = begin
    # set up the latent ode integration config
    Δt = LATENT_SPACE_SAMPLE_RATE_SCALER / m.sample_rate
    T = m.output_n * Δt

    return (
        # use the passed in rng for noise generation
        rng = rng,
        # states for all nested models in case they are recurrent
        feature_scanner = Lux.initialstates(rng, m.feature_scanner),
        control = Lux.initialstates(rng, m.control),
        encoder = Lux.initialstates(rng, m.encoder),
        identity_decoder = Lux.initialstates(rng, m.identity_decoder),
        decoder = Lux.initialstates(rng, m.decoder),
        ode = Lux.initialstates(rng, m.ode),
        # ode integration config
        saveat = 0f0:Δt:T-Δt,
        save_on = true,
        save_start = true,
        save_end = true,
        T = T-Δt,
        # a flag to track whether or not the input has been resampled
        resampled = false
    )
end

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
    tspan = (0.0f0, st.T)
    node = NeuralODE(m.ode, tspan, Tsit5(), saveat=st[:saveat] |> Lux.cpu, save_on=st[:save_on], save_start=st[:save_start], save_end=st[:save_end], verbose = false)
    control, _ = m.control(features, ps.control, st.control)

    ys, un = [
        begin
            ode_params = ComponentArray(view(control,:,i), m.ode_axes)
            ode_solution = node(view(u0,:,i), ode_params, st.ode)[1].u

            decoded_trajectory = m.decoder(reduce(hcat, ode_solution), ps.decoder, st.decoder)[1]
            noise = rand(st.rng, eltype(features), size(decoded_trajectory))

            new_u0 = ode_solution[:,end]
            
            (decoded_trajectory + noise)', new_u0
        end
        for i ∈ 1:size(features)[2]
    ] |> unzip

    return (cat([y' for y ∈ ys]..., dims=3), (hcat(un...),)), (st..., resampled = false)
end

fft_matrix(x) = begin
    ffts = [[Float32(y.re); Float32(y.im)] for y ∈ fft(x, 1)[1:end÷2, :]]
    ffts_as_matrices = [reduce(hcat, c) for c ∈ eachcol(ffts)]
    reshaped_ffts = reduce(A -> cat(A, dims = 3), ffts_as_matrices)
    reshaped_ffts
end

# bootleg broomhead-king embedding without normalization
function sound_svd(sound::Vector, d::Int)
    trajectory = trajectory_matrix(sound, d)
    res = LinearAlgebra.svd(trajectory)
    return res.U, res.S, res.Vt
end

trajectory_matrix(sound::Vector, d::Int) = Matrix(DynamicalSystems.embed(sound, d, 1))
