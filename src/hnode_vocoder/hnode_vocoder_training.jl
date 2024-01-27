include("hnode_vocoder.jl")
include("../nn/training_utils.jl")

using FLAC, FileIO, BSON, Dates, WAV

function create_synthesized_data(N, f0_floor, f0_ceil, sample_rate; harmonics = 1, output_dir="synthesized_data", loudness=0.2)
    sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    Δt = 1 / sample_rate
    T = N * Δt
    t = 0:Δt:T

    for f0 ∈ f0_floor:f0_ceil
        x = reduce(+, [sine_wave(f, 1, rand()).(t) for f ∈ (1:harmonics) .* f0])
        x = x ./ (rms(x) / loudness)
        filename = joinpath(output_dir, "$(f0)_x$(harmonics)_$(sample_rate).wav")
        wavwrite(x, filename; Fs = sample_rate)
    end
end

# load and resample all sounds for training from directories
function load_sounds(target_dirs...; file_limit_per_dir = nothing)
    result = Vector{Float32}()

    fade_duration = (FEATURE_EXTRACTION_SAMPLE_RATE * 0.1) |> Int # 100 millis
    w = DSP.Windows.hanning(2*fade_duration)
    fade_in = w[1:fade_duration]
    fade_out = w[fade_duration+1:end]

    for target_dir ∈ target_dirs
        println("loading files from $(target_dir)...")
        files = readdir(target_dir; join = true)
        
        dir_sounds = reduce(vcat, [
            begin
                if endswith(f, "flac") || endswith(f, "wav")
                    sound, sample_rate = load(f)
                    if sample_rate != FEATURE_EXTRACTION_SAMPLE_RATE
                        sound = resample(sound, FEATURE_EXTRACTION_SAMPLE_RATE / sample_rate) 
                    end
                    sound = sound[:,1]
                    sound[1:fade_duration] .*= fade_in
                    sound[end-fade_duration+1:end] .*= fade_out
                    sound / maximum(sound)
                else
                    Vector{Float32}()
                end
            end
            for f ∈ files[1:(isnothing(file_limit_per_dir) ? end : file_limit_per_dir)]
        ])

        result = vcat(result, dir_sounds)
    end

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

    return reduce(hcat, aggregated_sound), reduce(hcat, aggregated_features), reduce(hcat, fft_ys), output_n
end


function train_encoders(model::hNODEVocoder, ps, st::NamedTuple, series; batchsize = 1, chunk_step = 1, epochs = 1, optimiser = Optimisers.Adam())
    # data
    chunks = chunk_data([series;;], SVD_LATENT_DIMS; step=chunk_step)
    xs = reduce(hcat, chunks)
    ys = reduce(hcat, [(c' * model.sound_to_latent)' for c ∈ chunks])
    zs = reduce(hcat, [c[1] for c ∈ chunks])

    train_data = (xs, ys, zs)
    loader = DataLoader(train_data; batchsize = batchsize, shuffle = true)

    #training
    function loss(model, ps, st, data) 
        xs, ys, zs = data

        pred_ys, st_encoder = model.encoder(xs, ps.encoder, st.encoder)
        pred_zs, st_decoder = model.decoder(pred_ys, ps.decoder, st.decoder)

        loss_ys = sum(abs2, ys[1:10,:] .- pred_ys)
        loss_zs = sum(abs2, zs .- pred_zs)

        loss = loss_ys + loss_zs

        return loss, (st..., encoder = st_encoder, decoder = st_decoder), (pred_ys,)
    end

    cb(loss, epoch, ps, st) = begin
        overall_loss, _ = loss(model, ps, st, (xs, ys, zs))
        println("$(epoch): $(overall_loss)")
    end

    return train(model, loader, loss; ps = ps, st = st, epochs = epochs, optimiser = optimiser, epoch_cb = cb)
end

function train_nested_ode(
    hnode::hNODE,
    ps,
    st::NamedTuple,
    Δt,
    series;
    integration_length = 5,
    batchsize = 1,
    epochs = 100,
    cb = nothing,
    optimizer = Optimisers.Adam(0.001)
)
    # model
    T = integration_length * Δt
    tspan = (0.0, T)
    node = NeuralODE(hnode.hb, tspan, Tsit5(), saveat=Δt:Δt:T, save_on=true, save_start=false, save_end=true, verbose = true)
    hb_ps = ps.hb
    hb_st = st.hb

    #data
    target_trajectory = time_series_to_latent(hnode.encoder, ps.encoder, st.encoder, series, hnode.in_n)
    xs = target_trajectory[1:end-integration_length,:]' # ode_dimension x batch
    ys = cat([target_trajectory[i+1:i+integration_length,:] for i ∈ 1:size(xs)[2]]...; dims=3) # point_n x ode_dimension x batch
    loader = DataLoader((xs, ys); batchsize = batchsize, shuffle = true)

    #training
    function integrated_loss(model, ps, st, data)
        xs, ys = data
        solution, st_hb = model(xs, ComponentArray(ps), st) # point_n [ode_dimension x batch]
        loss = 0.0
        for i ∈ 1:length(solution.u)
            u = solution.u[i]
            y = ys[i,:,:]
            loss += sum(abs2, u .- y)
        end
        return loss, st_hb, ()
    end

    new_ps, new_st = train(node, loader, integrated_loss; ps = hb_ps, st = hb_st, cb = cb, epochs = epochs, optimiser = optimizer)

    return (ps..., hb = new_ps), (st..., hb = new_st)
end
