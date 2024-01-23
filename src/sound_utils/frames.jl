using WORLD
include("dio.jl")
include("cheaptrick.jl")

const MFCC_SIZE = 20

function get_features(frame::Vector{Float32}, dio::DIO, cheaptrick::CheapTrick, prev_f0_estimate = (0.0, 0.0))
    frame = frame |> Vector{Float64}
    # f0
    f0 = dio(frame; previous_estimate = prev_f0_estimate)
    # spectral part
    spectrogram = cheaptrick(f0[1], frame)
    mc = sp2mc(spectrogram, MFCC_SIZE, 0.41)[2:end]
    # aperiodicity
    aperiodicity = d4c(frame, dio.sample_rate, [0.0], [f0[1]])
    coded_aperiodicity = code_aperiodicity(aperiodicity, dio.sample_rate)
    # dynamics
    loudness = rms(frame)

    return loudness, f0, mc, coded_aperiodicity, spectrogram
end

function get_flat_features(frame, dio, cheaptrick)
    loudness, f0, mc, coded_aperiodicity, _ = get_features(frame, dio, cheaptrick)
    return [loudness; f0[1]; mc; coded_aperiodicity][:,1] |> Vector{Float32}
end

function sound_to_micro_frames(sound::Vector{Float32}, sample_rate::Int32; f0_floor = 60.0f0, f0_ceil = 600.0f0, freqs_cutoff::Float32=4000.0f0, hop = 1.0)

    spectrogram_size = nothing
    mc_size = nothing
    coded_aperiodicity_size = nothing

    dio = DIO(f0_ceil, f0_floor, sample_rate; relative_peak_threshold = 0.5, noise_floor = 0.005)
    cheaptrick = CheapTrick(f0_ceil, f0_floor, sample_rate)

    # the f0 estimator needs at least 1 / f0_floor samples to work properly so the frame size has to be lowered
    step = (hop * (sample_rate / 1000)) |> round |> Int
    frame_size = 3 * sample_rate ÷ f0_floor |> Int
    prev_f0_estimate = (0.0, 0.0)
    frames = [
        begin
            frame_range = if i+frame_size < length(sound)
                i:i+frame_size
            else
                i:-1:i-frame_size
            end
            loudness, f0, mc, coded_aperiodicity, spectrogram = get_features(sound[frame_range], dio, cheaptrick, prev_f0_estimate)
            features = [loudness; f0[1]; mc; coded_aperiodicity][:,1] |> Vector{Float32}
            target = reshape(spectrogram, length(spectrogram)) |> Vector{Float32}

            prev_f0_estimate = f0

            # sizes info
            spectrogram_size = isnothing(spectrogram_size) ? size(spectrogram) : spectrogram_size
            mc_size = isnothing(mc_size) ? size(mc) : mc_size
            coded_aperiodicity_size = isnothing(coded_aperiodicity_size) ? size(coded_aperiodicity) : coded_aperiodicity_size

            features, target
        end
        for i ∈ 1:step:length(sound)
    ]
    xs = hcat([f[1] for f in frames]...)
    ys = hcat([f[2] for f in frames]...)

    return (xs, ys), (mc_size = mc_size, coded_aperiodicity = coded_aperiodicity_size, spectrogram = spectrogram_size)
end


function sound_to_micro_frames_WORLD(sound::Vector{Float32}, sample_rate::Int32; f0_floor = 60.0f0, f0_ceil = 600.0f0, freqs_cutoff::Float32=4000.0f0)

    spectrogram_size = nothing
    mc_size = nothing
    coded_aperiodicity_size = nothing

    # the f0 estimator needs at least 1 / f0_floor samples to work properly so the frame size has to be lowered
    step = sample_rate * 0.001 |> round |> Int
    frame_size = 2 * sample_rate ÷ f0_floor |> Int
    frames = [
        begin
            frame_range = i:i+frame_size
            frame = sound[frame_range] |> Vector{Float64}

            # f0
            f0, = dio(frame, sample_rate; f0_floor=f0_floor, f0_ceil=f0_ceil)

            # spectral part
            @time spectrogram = cheaptrick(frame, sample_rate, [0.0], [f0])
            mc = sp2mc(spectrogram, 20, 0.41)

            # aperiodicity
            aperiodicity = d4c(frame, sample_rate, [0.0], [f0])
            coded_aperiodicity = code_aperiodicity(aperiodicity, sample_rate)

            # dynamics
            loudness = rms(frame)

            # flattened vector of features and target
            features = [sample_rate; loudness; f0; mc; coded_aperiodicity][:,1] |> Vector{Float32}
            target = reshape(spectrogram, length(spectrogram)) |> Vector{Float32}

            # sizes info
            spectrogram_size = isnothing(spectrogram_size) ? size(spectrogram) : spectrogram_size
            mc_size = isnothing(mc_size) ? size(mc) : mc_size
            coded_aperiodicity_size = isnothing(coded_aperiodicity_size) ? size(coded_aperiodicity) : coded_aperiodicity_size

            features, target
        end
        for i ∈ 1:step:(length(sound)-frame_size)
    ]
    xs = hcat([f[1] for f in frames]...)
    ys = hcat([f[2] for f in frames]...)

    return (xs, ys), (mc_size = mc_size, coded_aperiodicity = coded_aperiodicity_size, spectrogram = spectrogram_size)
end

# the dimensions of the sequences are ((x/y)_len, seq_len, batch_len)
# the model will iterate over the second dimension, processing multiple batches in parallel
function frames_to_data(xs, ys, seq_len; batch_size = nothing, seq_overlap = nothing)
    step = if isnothing(seq_overlap) max(seq_len ÷ 10, 1) else seq_overlap end
    last_index = size(xs)[end] - seq_len + 1

    xs_seqenced = [view(xs, :, i:i+seq_len-1) for i ∈ 1:step:last_index]
    ys_seqenced = [view(ys, :, i:i+seq_len-1) for i ∈ 1:step:last_index]

    batch_size = if isnothing(batch_size) size(xs_seqenced)[1] else batch_size end

    return DataLoader((xs_seqenced, ys_seqenced), batch_size), batch_size
end
