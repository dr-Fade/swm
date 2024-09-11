include("hnode_vocoder.jl")
include("../nn/training_utils.jl")
include("../sound_utils/sound_file_utils.jl")

using FLAC, FileIO, BSON, Dates, WAV, Plots, Distributed

function get_training_data(model::hNODEVocoder, sound; β=0f0, drop_mfccs=false)
    frame_size = model.n

    feature_scanner = model.feature_scanner
    stream_filter = model.stream_filter

    rng = Random.default_rng()
    stream_filter_ps, stream_filter_st = Lux.setup(rng, stream_filter)
    target_stream_filter_st = deepcopy(stream_filter_st)
    feature_scanner_ps, feature_scanner_st = Lux.setup(rng, feature_scanner)

    aggregated_input_sound = Vector{Vector{Float32}}()
    aggregated_target_sound = Vector{Vector{Float32}}()
    aggregated_features = Vector{Vector{Float32}}()

    for i ∈ 1:frame_size:length(sound)-frame_size
        frame = sound[i:i+frame_size-1]
        frame_with_noise = (1-β)*frame + β*randn(Float32, length(frame))

        filtered_frame, target_stream_filter_st = stream_filter(frame, stream_filter_ps, target_stream_filter_st)
        filtered_frame_with_noise, stream_filter_st = stream_filter(frame_with_noise, stream_filter_ps, stream_filter_st)
        features, feature_scanner_st = feature_scanner([filtered_frame;;], feature_scanner_ps, feature_scanner_st)

        push!(aggregated_input_sound, filtered_frame_with_noise[:])
        push!(aggregated_target_sound, filtered_frame[1:model.n])
        push!(aggregated_features, features[:])
    end

    return (
        input=hcat(aggregated_input_sound...),
        target=hcat(aggregated_target_sound...),
        features=hcat(aggregated_features...)
    )
end

function get_denoising_training_data(model::hNODEVocoder; noise_length=FEATURE_EXTRACTION_SAMPLE_RATE, max_noise_level=0.001f0)
    sound = max_noise_level*randn(Float32, noise_length)
    training_data = get_training_data(model, sound; β=0f0)
    return (training_data..., target=zeros(Float32, size(training_data.target)))
end

function get_connected_trajectory(model, ps, st, data)
    frames, features = data

    output_n = model.n
    Δt = LATENT_SPACE_SAMPLE_RATE_SCALER / model.sample_rate
    T = output_n * Δt

    st = (st..., saveat = 0f0:Δt:T, T = T, resampled = true) 

    u0s = model.encoder(frames[:,1:1], ps.encoder, st.encoder)[1]
    res = Vector{Float32}()
    loss = 0
    for i ∈ 1:size(frames)[2]
        (pred_ys, (u0s,)), st = model((frames[:,i:i], (u0s, features[:,i:i])), ps, st)
        append!(res, pred_ys[:,1])
        loss += sum(abs2, pred_ys .- frames[1:output_n,i])
    end
    return res, loss
end

psum(f, xs) = begin
    ch = Channel(Threads.nthreads())
    res = 0
    @sync begin
        for x ∈ xs
            Threads.@spawn put!(ch, f(x))
        end
        for x ∈ xs
            res += take!(ch)
        end
    end
    return res
end

function train_final_model(
    model::hNODEVocoder, ps, st::NamedTuple, training_data;
    batchsize = 1, slices = 1, epochs = 1, epoch_cb = nothing, batch_cb = nothing, optimiser = Optimisers.Adam(), distributed_backend = nothing
)
    output_n = model.n

    st = (st..., resampled = true)

    frames = training_data.input
    target = training_data.target
    features = training_data.features
    N = size(frames)[end]
    frames_batches = Array{Float32, 3}(undef, (size(frames)[1], slices, N-slices))
    target_batches = Array{Float32, 3}(undef, (size(target)[1], slices, N-slices))
    features_batches = Array{Float32, 3}(undef, (size(features)[1], slices, N-slices))

    for i in 1:N-slices
        frames_batches[:,:,i] = view(frames, :, i:i+slices-1)
        target_batches[:,:,i] = view(target, :, i:i+slices-1)
        features_batches[:,:,i] = view(features, :, i:i+slices-1)
    end
    data = (frames_batches, target_batches, features_batches)
    loader = DataLoader(data; batchsize = batchsize, shuffle=true)

    window = blackman(output_n*slices)
    #training
    function loss_function(model, ps, st, data)
        input, target, features = data

        u0s = model.encoder(input[:,1,:], ps.encoder, st.encoder)[1]

        target_sound = vcat(eachslice(target[1:output_n,:,:]; dims = 2)...)
        generated_sound = Matrix{Float32}(undef, 0, size(target_sound)[2])

        encoder_loss = 0f0
        for (input_slice, features_slice) ∈ zip(eachslice(input; dims = 2), eachslice(features; dims = 2))
            (pred_ys, (u0s,)), st = model((input_slice, (u0s, features_slice)), ps, st)
            generated_sound = vcat(generated_sound, pred_ys)

            encoder_loss += sum(abs2, input_slice[1,:]' - model.decoder(model.encoder(input_slice, ps.encoder, st.encoder)[1], ps.decoder, st.decoder)[1])
        end

        regularization_loss = 0f0
        raw_loss = sum(abs2, generated_sound - target_sound)
                 + sum(abs, fft(generated_sound) .- fft(target_sound))
                 + sum(abs2, log10.(abs.(fft(generated_sound .* window)[1:end÷2] .+ eps(Float32)))
                          .- log10.(abs.(fft(target_sound .* window)[1:end÷2] .+ eps(Float32))))

        return regularization_loss + (encoder_loss + raw_loss) / slices / batchsize, st, (nothing,)
    end

    ps, st = train(
        model, loader, loss_function;
        ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb, batch_cb = batch_cb, states_to_clear = (:carry, :prev_features), distributed_backend = distributed_backend
    )

    return ps, st
end
