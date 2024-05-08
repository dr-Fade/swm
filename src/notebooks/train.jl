include("../hnode_vocoder/hnode_vocoder_training.jl")

model = hNODEVocoder(FEATURE_EXTRACTION_SAMPLE_RATE; speaker = false)

rng = Random.default_rng()
model_filename = "hnode_vocoder_params.bson"
if isfile(model_filename)
    BSON.@load model_filename ps
    @info "loading params from file: $model_filename"
    st = Lux.initialstates(rng, model)
else
    @info "creating new params"
    ps, st = Lux.setup(rng, model)
end

sound_dirs = []

dev_voice_sounds = "../samples/LibriSpeech/dev-clean/84"
synthetic_data = "../samples/synthesized_sounds"

for (root, dirs, files) ∈ walkdir(synthetic_data)
    for d ∈ dirs
        dir_path = joinpath(root, d)
        nested_files = readdir(dir_path)
        if any(endswith.(nested_files, ".flac")) || any(endswith.(nested_files, ".wav"))
            push!(sound_dirs, dir_path)
        end
    end
    if any(endswith.(files, ".flac")) || any(endswith.(files, ".wav"))
        push!(sound_dirs, root)
    end
end

test_sound, sample_rate = load_sounds(sound_dirs...; file_limit_per_dir = nothing)
frames, features = get_training_data(model, vcat(test_sound..., zeros(model.input_n)), FEATURE_EXTRACTION_SAMPLE_RATE)

file_n = length(test_sound)-1

file_duration = 3600 ÷ 16
start_index = 1
start_frame = start_index * file_duration
N = file_n * file_duration
plot(features[2,start_frame:start_frame+N] .* feature_stds[2] .+ feature_means[2], size=(1000,500)) |> display
heatmap(features[:,start_frame:start_frame+N], size=(1000,500)) |> display

start_index_test = 1 * file_duration
M = 2 * file_duration
sound = vcat(eachcol(frames[1:16,start_index_test:start_index_test+M-1])...)

function cb(loss_function, epoch, model, ps, st)
    if any(isnan, ComponentArray(ps))
        error("NAN encountered")
    end

    latent_xs = time_series_to_latent(model.encoder, ps.encoder, st.encoder, sound, model.input_n)
    decoded = latent_to_time_series(model.decoder, ps.decoder, st.decoder, latent_xs)
    ys, _ = get_connected_trajectory(
        model, ps, st,
        (frames[:,start_index_test:start_index_test+M+1],
        features[:,start_index_test:start_index_test+M+1])
    )
    plot(
        plot(latent_xs[:, 1], latent_xs[:, 2], latent_xs[:, 5]),
        plot([sound, decoded, ys], label=["sound" "decoded" "model output"]),
        heatmap(ps.encoder.layer_2.weight'),
        size = (800, 600),
        layout = @layout [[a;b] c]
    ) |> display
    common_length = min(length(ys), length(sound))
    l = sum(abs2, sound[1:common_length] - ys[1:common_length])
    display(l)
    @save "hnode_vocoder_params.bson" ps
    return false
end

ps, st = train_final_model(model, ps, st, (frames[:,start_index:start_index+N], features[:,start_index:start_index+N]); batchsize = 2, slices = 1, optimiser = Optimisers.ADAM(1e-4), epochs = 1, cb = cb)