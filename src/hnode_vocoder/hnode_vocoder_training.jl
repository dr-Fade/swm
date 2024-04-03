include("hnode_vocoder.jl")
include("../nn/training_utils.jl")

using FLAC, FileIO, BSON, Dates, WAV, Plots

function create_synthesized_data(N, f0_floor, f0_ceil, sample_rate; harmonics = 1, output_dir="synthesized_data", max_peak=1.0, f_step=5)
    sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    Δt = 1 / sample_rate
    T = N * Δt
    t = 0:Δt:T

    for f0 ∈ f0_floor:f_step:f0_ceil
        x = reduce(+, [sine_wave(f, 1, rand()).(t) for f ∈ (1:harmonics) .* f0])
        x = max_peak * (x ./ maximum(x))
        filename = if max_peak < 1.0
            joinpath(output_dir, "$(f0)_x$(harmonics)_$(max_peak)_$(sample_rate).wav")
        else
            joinpath(output_dir, "$(f0)_x$(harmonics)_$(sample_rate).wav")
        end
        wavwrite(x, filename; Fs = sample_rate)
    end
end

# load and resample all sounds for training from directories
function load_sounds(target_dirs...; file_limit_per_dir = nothing)
    fade_duration = (FEATURE_EXTRACTION_SAMPLE_RATE * 0.1) |> Int # 100 millis
    w = DSP.Windows.hanning(2*fade_duration)
    fade_in = w[1:fade_duration]
    fade_out = w[fade_duration+1:end]

    result = vcat([
        begin
            # println("loading files from $(target_dir)...")
            files = readdir(target_dir; join = true)

            [
                begin
                    if endswith(f, "flac") || endswith(f, "wav")
                        sound, sample_rate = load(f)
                        if sample_rate != FEATURE_EXTRACTION_SAMPLE_RATE
                            sound = resample(sound, FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate) 
                        end
                        sound = sound[:,1]
                        sound[1:fade_duration] .*= fade_in
                        sound[end-fade_duration+1:end] .*= fade_out
                        Vector{Float32}(sound)
                    else
                        Vector{Float32}()
                    end
                end
                for f ∈ files[1:(isnothing(file_limit_per_dir) ? end : file_limit_per_dir)]
            ]
        end
        for target_dir ∈ target_dirs
    ]...)

    return result, FEATURE_EXTRACTION_SAMPLE_RATE
end

function get_training_data(sound, sample_rate; output=nothing)
    sound = resample(sound, FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate; dims = 1)

    dio = DIO(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE)
    cheaptrick = CheapTrick(F0_CEIL, F0_FLOOR, FEATURE_EXTRACTION_SAMPLE_RATE)

    frame_size = max(dio.frame_length, cheaptrick.max_frame_size) # i.e. the input size of the model
    hop_size = FEATURE_EXTRACTION_SAMPLE_RATE ÷ 1000 # 1 millisecond

    output_n = FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_FLOOR |> Int # contain the F0_FLOOR period

    fft_size = nextfastfft(2 ^ ceil(log2(output_n)) |> Int)
    hanning_window = hanning(output_n)

    feature_scanner = FeatureScanner(dio, cheaptrick) |> Lux.StatefulRecurrentCell
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, feature_scanner)

    aggregated_sound = Vector{Vector{Float32}}()
    aggregated_features = Vector{Vector{Float32}}()
    fft_ys = Vector{Vector{ComplexF32}}()
    for i ∈ 1:hop_size:length(sound)-frame_size
        frame = sound[i:i+frame_size-1]
        push!(aggregated_sound, frame)

        features, st = feature_scanner([frame;;], ps, st)
        push!(aggregated_features, features[:])

        expected_fft = frame[1:output_n] .* hanning_window |> x -> [x; zeros(fft_size - output_n)] |> fft
        push!(fft_ys, expected_fft)
    end

    return hcat(aggregated_sound...), hcat(aggregated_features...), hcat(fft_ys...), output_n
end

function train_encoder(model::hNODEVocoder, ps, st::NamedTuple, chunks; batchsize = 1, epochs = 1, optimiser = Optimisers.Adam())
    autoencoder = Lux.Chain(;
        encoder = model.encoder,
        decoder = model.identity_decoder
    )
    autoencoder_ps = (encoder = ps.encoder, decoder = ps.identity_decoder)
    autoencoder_st = (encoder = st.encoder, decoder = st.identity_decoder)

    loader = DataLoader((chunks,); batchsize = batchsize, shuffle = true)

    function loss(model, ps, st, data)
        x, = data
        y = x
        ŷ = model(x, ps, st)[1]
        loss = sum(abs2, y .- ŷ) / length(y)
        return loss, st, ()
    end

    autoencoder_ps, autoencoder_st = train(autoencoder, loader, loss; ps = autoencoder_ps, st = autoencoder_st, epochs = epochs, optimiser = optimiser)

    return (ps..., encoder = autoencoder_ps.encoder, identity_decoder = autoencoder_ps.decoder),
            (st..., encoder = autoencoder_st.encoder, identity_decoder = autoencoder_st.decoder)
end

function train_decoder(model::hNODEVocoder, ps, st::NamedTuple, chunks; batchsize = 1, epochs = 1, optimiser = Optimisers.Adam())
    encoder, encoder_ps, encoder_st = get_encoder(model, ps, st)

    xs = encoder(chunks, encoder_ps, encoder_st)[1]
    ys = chunks[1,:]
    loader = DataLoader((xs,ys); batchsize = batchsize, shuffle = true)

    function loss(model, ps, st, data)
        x, y = data
        ŷ, st = model(x, ps, st)
        loss = sum(abs2, y .- ŷ) / length(y)
        return loss, st, ()
    end

    decoder_ps, decoder_st = train(model.decoder, loader, loss; ps = ps.decoder, st = st.decoder, epochs = epochs, optimiser = optimiser)

    return (ps..., decoder = decoder_ps), (st..., decoder = decoder_st)
end

function train_encoders(model::hNODEVocoder, ps, st::NamedTuple, series; batchsize = 1, epochs = 1, optimiser = Optimisers.Adam())
    chunks = (embed(series, ENCODER_INPUT_N, 1) |> Matrix)'
    filtered_chunks = hcat(filter(x -> rms(x) > 0.1, eachcol(chunks))...)
    ps, st = train_encoder(model, ps, st, filtered_chunks; batchsize = batchsize, epochs = epochs, optimiser = optimiser)
    return train_decoder(model, ps, st, filtered_chunks; batchsize = 1, epochs = epochs, optimiser = optimiser)
end

function integrate_nested_ode(model::hNODEVocoder, node_ps, node_st, u0; integration_n = nothing)
    integration_n = isnothing(integration_n) ? model.output_n : integration_n
    Δt = LATENT_SPACE_SAMPLE_RATE_SCALER / model.sample_rate
    T = (integration_n-1) * Δt
    saveat = 0f0:Δt:T
    tspan = (0f0, T)

    node = NeuralODE(model.ode, tspan, Tsit5(), saveat=saveat, save_on=true, save_start=true, save_end=true, verbose = false)
    if isnothing(node_st)
        rng = Random.default_rng()
        node_st = Lux.initialstates(rng, node)
    end

    return node(u0, node_ps, node_st)
end

function get_target_parameters(
    hnode_vocoder, hnode_vocoder_ps, hnode_vocoder_st, series_filename, res_dir;
    max_series_length = nothing, integration_n = 10, init_ps = nothing, epochs = 1, batchsize = 1, optimiser = Optimisers.Adam(),
    target_loss_threshold = 1.0
)
    if !isdir(res_dir)
        mkdir(res_dir)
    end

    series_node_ps_filename = begin
        f = basename(series_filename)
        dot_id = findlast(==('.'), f)
        output_target = joinpath(res_dir, f[1:dot_id-1])
        "$(output_target).bson"
    end

    node_ps = if !isnothing(init_ps)
        @info "using the passed in init_ps as initial parameters"
        init_ps
    elseif isfile(series_node_ps_filename)
        @info "using initial parameters in $(basename(series_node_ps_filename))"
        @load series_node_ps_filename ps
        ps
    else
        @info "no initial parameters provided for $(series_filename), will start training from scratch"
        # error("no initial parameters provided for $(series_filename), will start training from scratch")
    end

    sound, sample_rate = load(series_filename)
    # resample and truncate
    if sample_rate != FEATURE_EXTRACTION_SAMPLE_RATE
        sound = resample(sound, FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate) 
    end
    sound = sound[1:(isnothing(max_series_length) ? length(sound) : max_series_length), 1]

    epoch_cb(loss, epoch, model, ps, st) = begin
        l, _ = loss(model, ps, st, (1:10,))
        @save series_node_ps_filename ps
        println("[$(basename(series_filename))] $(epoch): $(l)")
        return l > target_loss_threshold
    end

    return _get_target_parameters(hnode_vocoder, hnode_vocoder_ps, hnode_vocoder_st, sound[:,1]; epochs = epochs, batchsize = batchsize, epoch_cb=epoch_cb, integration_n = integration_n, optimiser = optimiser, init_node_ps = node_ps)
end

function plot_node_trajectories_for_data(
    hnode_vocoder, hnode_vocoder_ps, hnode_vocoder_st, series_filename, res_dir;
    max_series_length = nothing
)
    series_node_ps_filename, plot_filename = begin
        f = basename(series_filename)
        dot_id = findlast(==('.'), f)
        output_target = joinpath(res_dir, f[1:dot_id-1])
        ps_filename = "$(output_target).bson"
        plot_filename = "$(output_target).png"
        ps_filename, plot_filename
    end

    node_ps = if isfile(series_node_ps_filename)
        @load series_node_ps_filename ps
        ps
    else
        error("no initial parameters provided for $(series_filename)")
    end

    sound, sample_rate = load(series_filename)
    # resample and truncate
    if sample_rate != FEATURE_EXTRACTION_SAMPLE_RATE
        sound = resample(sound, FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate) 
    end
    sound = sound[1:(isnothing(max_series_length) ? length(sound) : max_series_length), 1]
    
    target_trajectory = time_series_to_latent(hnode_vocoder.encoder, hnode_vocoder_ps.encoder, hnode_vocoder_st.encoder, sound, ENCODER_INPUT_N)
    predicted_trajectory, _ = integrate_nested_ode(hnode_vocoder, node_ps, nothing, target_trajectory[1,:]; integration_n = 2*model.output_n)
    plt = plot(
        (plot([target_trajectory[1:800,i], predicted_trajectory[i,:]], labels = ["target" "prediction"]) for i ∈ 1:10)...,
        size = (900, 2000),
        layout = Plots.@layout[a b ; c d ; e f; g h ; i j]
    )
    savefig(plt, plot_filename)
end

function _get_target_parameters(
    model::hNODEVocoder,
    ps,
    st::NamedTuple,
    series;
    batchsize = 1,
    epochs = 1,
    integration_n = nothing,
    epoch_cb = nothing,
    optimiser = Optimisers.Adam(),
    init_node_ps = nothing
)
    #model
    integration_n = isnothing(integration_n) ? model.output_n : integration_n
    Δt = LATENT_SPACE_SAMPLE_RATE_SCALER / model.sample_rate
    T = (integration_n-1) * Δt
    saveat = 0f0:Δt:T
    tspan = (0f0, T)

    node = NeuralODE(model.ode, tspan, Tsit5(), saveat=saveat, save_on=true, save_start=true, save_end=true, verbose = false)
    ode_st = Lux.initialstates(st.rng, node)

    ode_ps = if isnothing(init_node_ps)
        Lux.initialparameters(st.rng, node)
    else
        init_node_ps
    end

    #data
    target_trajectory = time_series_to_latent(model.encoder, ps.encoder, st.encoder, series, ENCODER_INPUT_N)'
    start_ids = 1 : integration_n÷4 : size(target_trajectory)[2]-integration_n
    loader = DataLoader((start_ids,); batchsize = batchsize, shuffle = true)
    rng = st.rng

    #training
    function loss(model, ps, st, data)
        ids, = data
        loss = 0f0
        ca = ComponentArray(ps)
        for i ∈ ids
            x = target_trajectory[:,i]
            perturbed_x = x# + 0.0001*(rand(rng, Float32, size(x)) .- 0.5)
            y = target_trajectory[:,i:i+integration_n-1]
            ŷ, st = model(perturbed_x, ca, st)
            loss += sum(abs2, y .- ŷ)
        end
        # eigens = real.(LinearAlgebra.eigvals(ps[:linear_block][:weight]))
        # return loss/length(ids) + sum(abs, ca)/length(ca) + sum(exp, eigens), st, ()
        return loss/length(ids)/integration_n, st, ()
    end

    ode_ps, _ = train(node, loader, loss; ps = ode_ps, st = ode_st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb)

    return ode_ps
end

function train_final_model(model::hNODEVocoder, ps, st::NamedTuple, series; batchsize = 1, epochs = 1, integration_n = nothing, epoch_cb = nothing, optimiser = Optimisers.Adam())
    init_st = st
    integration_n = isnothing(integration_n) ? model.output_n : integration_n

    Δt = LATENT_SPACE_SAMPLE_RATE_SCALER / model.sample_rate
    T = integration_n * Δt

    st = (st..., saveat = 0f0:Δt:T-Δt, T = T-Δt, resampled = true) 

    #data
    sound_regions, features = get_training_data(series, FEATURE_EXTRACTION_SAMPLE_RATE)
    u0s = model.encoder(sound_regions, ps.encoder, st.encoder)[1]
    truncated_sound_regions = sound_regions[1:integration_n,:]
    data = ((sound_regions, (u0s, features)), reshape(truncated_sound_regions, 1, size(truncated_sound_regions)...))

    loader = DataLoader(data; batchsize = batchsize, shuffle = true)

    #training
    function loss(model, ps, st, data)
        xs, ys = data
        (pred_ys, _), st = model(xs, ps, st)
        loss = sum(abs2, ys .- pred_ys)
        return loss, st, (pred_ys,)
    end
    ps, _ = train(model.control, loader, loss; ps = ps.control, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = epoch_cb)
    return ps, init_st
end
