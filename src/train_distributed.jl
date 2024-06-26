using Pkg

Pkg.instantiate(); Pkg.precompile();

using Lux, MPI

# init the mpi backend
DistributedUtils.initialize(MPIBackend)

backend = DistributedUtils.get_distributed_backend(MPIBackend)

const local_rank = DistributedUtils.local_rank(backend)
const total_workers = DistributedUtils.total_workers(backend)
const worker_id = "$(readchomp(`hostname`)):$local_rank"

function log(msg)
    suffix = local_rank == 0 ? " [ROOT]" : ""
    println("$worker_id$suffix: $msg")
end

function root_log(msg)
    local_rank == 0 && log(msg)
end

root_log("initialized $total_workers workers")

# load the model and parameters
include("hnode_vocoder/hnode_vocoder_training.jl");

model = hNODEVocoder(FEATURE_EXTRACTION_SAMPLE_RATE; n = 2*Integer(FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_CEIL), speaker = false)

rng = Random.default_rng()
model_filename = ARGS[1]
if isfile(model_filename)
    BSON.@load model_filename ps
    root_log("loading params from file: $model_filename")
    st = Lux.initialstates(rng, model)
else
    root_log("creating new params")
    ps, st = Lux.setup(rng, model)
end

# training and test data dirs
# TODO: replace with cli args parsing
dev_voice_sounds = "src/samples/LibriSpeech/dev-clean"
synthetic_data = "src/samples/synthesized_sounds"
test_data_dir = "src/samples/LibriSpeech/test"

# load files on the root worker and broadcast to everyone else
sounds = MPI.bcast(
    load_sounds(dev_voice_sounds; target_sample_rate=FEATURE_EXTRACTION_SAMPLE_RATE, file_limit_per_dir = 1, verbose = local_rank==0, shuffle_files = true),
    backend.comm
)

root_log("loaded $(length(sounds)) files in total")

# filter out the files based on the local rank of each worker
sounds = sounds[local_rank+1:total_workers:end]

log("picked $(length(sounds)) files")

concatenated_sounds = vcat([vcat(zeros(Float32, 1000), s) for s in sounds]...)
N = MPI.Reduce(length(concatenated_sounds), min, backend.comm)
N = MPI.bcast(N, backend.comm)

concatenated_sounds = concatenated_sounds[1:N]

training_data = get_training_data(model, concatenated_sounds, FEATURE_EXTRACTION_SAMPLE_RATE; β = 0.005f0)
synthetic_training_data = begin
    st_sounds = load_sounds(synthetic_data; target_sample_rate=FEATURE_EXTRACTION_SAMPLE_RATE, verbose = local_rank==0, shuffle_files = true)
    st_concatenated_sounds = vcat([vcat(zeros(Float32, 1000), s) for s in st_sounds]...)
    get_training_data(model, st_concatenated_sounds, FEATURE_EXTRACTION_SAMPLE_RATE; β = 0.005f0)
end
pitch_emphasis_training_data = begin
    pe_sounds = load_sounds(synthetic_data; target_sample_rate=FEATURE_EXTRACTION_SAMPLE_RATE, file_limit_per_dir = 1, verbose = false, shuffle_files = true)
    pe_concatenated_sounds = vcat([vcat(zeros(Float32, 1000), s) for s in pe_sounds]...)
    get_training_data(model, pe_concatenated_sounds, FEATURE_EXTRACTION_SAMPLE_RATE; drop_mfccs=true)
end
denoising_training_data = get_denoising_training_data(model; noise_length=max(length(concatenated_sounds)÷10, 1024), max_noise_level=0.005f0)

log("loaded $(size(training_data.input)[2]) training samples")
log("loaded $(size(synthetic_training_data.input)[2]) synthetic samples")
log("loaded $(size(pitch_emphasis_training_data.input)[2]) pitch emphasis training samples")
log("loaded $(size(denoising_training_data.input)[2]) denoising training samples")

training_data = (
    input = hcat(training_data.input, synthetic_training_data.input, pitch_emphasis_training_data.input, denoising_training_data.input),
    target = hcat(training_data.target, synthetic_training_data.target, pitch_emphasis_training_data.target, denoising_training_data.target),
    features = hcat(training_data.features, synthetic_training_data.features, pitch_emphasis_training_data.features, denoising_training_data.features)
)

test_data = nothing
test_sound = nothing

if local_rank == 0
    test_data = get_training_data(model, vcat(load_sounds(test_data_dir)...), FEATURE_EXTRACTION_SAMPLE_RATE)
    test_sound = vcat(eachcol(test_data.target[1:model.n,:])...)
    log("loaded $(size(test_data.target)[2]) test samples")
end

function epoch_cb(_, epoch, epochs, model, ps, st)
    if local_rank != 0
        return false
    end
    if any(isnan, ComponentArray(ps))
        error("NAN encountered")
    end
    l = -1
    try
        # latent_xs = time_series_to_latent(model.encoder, ps.encoder, st.encoder, test_sound, model.stream_filter.cell.n)
        # decoded = latent_to_time_series(model.decoder, ps.decoder, st.decoder, latent_xs)
        # ys, _ = get_connected_trajectory(model, ps, st, (test_data.input, test_data.features))

        # plt = plot(
        #     plot(eachcol(latent_xs)),
        #     plot([test_sound, decoded, ys], label=["test_sound" "decoded" "model output"]),
        #     heatmap(ps.encoder.layer_3.weight'),
        #     size = (1200, 800),
        #     layout = @layout [[a; b] c]
        # )

        # l = sum(abs2, test_sound .- ys)

        # savefig(plt, "training_results/$(now())_$(l).png")
        @save "training_results/$(now())_hnode_vocoder_params.bson" ps
    catch e 
        @warn e
    finally
        open("training_log.txt","a") do file
            println(file,"$(now()): epoch $epoch/$epochs, loss $(round(l; digits=2))")
        end
    end
    return false
end

function batch_cb(loss_function, model, ps, st, batch_i, batch_n, epoch)
    if local_rank != 0
        return false
    end
    if any(isnan, ComponentArray(ps))
        error("NAN encountered")
    end
    if batch_i == batch_n || batch_i % 100 == 0
        log_str = "$(now()): epoch $epoch, batch $batch_i/$batch_n"
        root_log(log_str)
        @save model_filename ps
        open("training_log.txt","a") do file
            println(file, log_str)
        end
    end

    return false
end

ps, st = train_final_model(
    model, ps, st,
    training_data;
    batchsize = 64,
    slices = 2,
    optimiser = Optimisers.ADAM(1e-5),
    epochs = 2,
    epoch_cb = epoch_cb,
    batch_cb = batch_cb,
    distributed_backend = backend
)
