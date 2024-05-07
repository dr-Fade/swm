include("hnode_vocoder.jl")
include("../nn/training_utils.jl")
include("../sound_utils/sound_file_utils.jl")

using FLAC, FileIO, BSON, Dates, WAV, Plots, Distributed

function get_training_data(model::hNODEVocoder, sound, sample_rate; output=nothing)
    sound = resample(sound, Float32(FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate); dims = 1) |> Vector{Float32}

    frame_size = model.input_n
    hop_size = model.output_n

    feature_scanner = model.feature_scanner
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, feature_scanner)

    aggregated_sound = Vector{Vector{Float32}}()
    aggregated_features = Vector{Vector{Float32}}()

    for i ∈ 1:hop_size:length(sound)-frame_size
        frame = sound[i:i+frame_size-1]
        push!(aggregated_sound, frame)

        features, st = feature_scanner([frame;;], ps, st)
        push!(aggregated_features, features[:])
    end

    return hcat(aggregated_sound...), hcat(aggregated_features...)
end

function get_connected_trajectory(model, ps, st, data)
    frames, features = data
    output_n = model.output_n
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

function train_final_model(model::hNODEVocoder, ps, st::NamedTuple, training_data; batchsize = 1, slices = 1, epochs = 1, epoch_cb = nothing, batch_cb = nothing, optimiser = Optimisers.Adam())
    output_n = model.output_n

    st = (st..., resampled = true)

    frames, features = training_data
    N = size(frames)[2]
    #data
    data = (
        reshape(frames[:,1:end-N%slices], size(frames)[1], slices, N÷slices),
        reshape(features[:,1:end-N%slices], size(features)[1], slices, N÷slices)
    )
    loader = DataLoader(data; batchsize = batchsize, shuffle=true)

    window = blackman(model.output_n*slices)
    #training
    function loss_function(model, ps, st, data)
        frames, features = data

        u0s = model.encoder(frames[:,1,:], ps.encoder, st.encoder)[1]

        target_sound = vcat(eachslice(frames[1:output_n,:,:]; dims = 2)...)
        generated_sound = Matrix{Float32}(undef, 0, size(target_sound)[2])

        regularization_loss = 0f0
        for (frames_slice, features_slice) ∈ zip(eachslice(frames; dims = 2), eachslice(features; dims = 2))
            (pred_ys, (u0s,)), st = model((frames_slice, (u0s, features_slice)), ps, st)
            generated_sound = vcat(generated_sound, pred_ys)
            regularization_loss += sum(abs, model.control(features_slice, ps.control, st.control)[1])
        end
        regularization_loss /= Lux.parameterlength(model.ode)

        raw_loss = sum(abs2, 100*(generated_sound .- target_sound)) / slices

        generated_fft = log10.((abs2.(fft(generated_sound .* window, 1)[1:end÷2,:]) .+ eps()))
        target_fft = log10.((abs2.(fft(target_sound .* window, 1)[1:end÷2,:]) .+ eps()))

        spectral_loss = sum(abs2, generated_fft .- target_fft)

        return (regularization_loss + raw_loss + spectral_loss) / batchsize, st, (nothing,)
    end

    ps, st = train(model, loader, loss_function; ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb, batch_cb = batch_cb, states_to_clear = (:carry, :prev_features))

    return ps, st
end
