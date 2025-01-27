using Pkg

Pkg.instantiate(); Pkg.precompile(io=devnull);

using Lux, MPI

# init the mpi backend
DistributedUtils.initialize(MPIBackend)

backend = DistributedUtils.get_distributed_backend(MPIBackend)

const BATCHES_BETWEEN_CHECKPOINTS = 10

const local_rank = DistributedUtils.local_rank(backend)
const total_workers = DistributedUtils.total_workers(backend)
const worker_id = "$local_rank"

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

model = hNODEVocoder(FEATURE_EXTRACTION_SAMPLE_RATE; n = Integer(FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_CEIL) ÷ 3)

while true

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
dev_voice_files = repeat(get_all_sound_files("src/samples/LibriSpeech/demo"), 20)
synthetic_files = shuffle(get_all_sound_files("src/samples/synthesized_sounds"))[1:20]

log("picked $(length(synthetic_files) + length(dev_voice_files)) files")

# truncate the total sound length to ensure all workers finish at the same time
dev_voice_sounds = get_sound(dev_voice_files)
N = MPI.Reduce(length(dev_voice_sounds), min, backend.comm)
N = MPI.bcast(N, backend.comm)
dev_voice_sounds = dev_voice_sounds[1:N]

synthetic_sounds = get_sound(synthetic_files)
N = MPI.Reduce(length(synthetic_sounds), min, backend.comm)
N = MPI.bcast(N, backend.comm)
synthetic_sounds = synthetic_sounds[1:N]

training_data = get_training_data(model, dev_voice_sounds)
synthetic_training_data = get_training_data(model, synthetic_sounds; β=0.01f0)
denoising_training_data = get_denoising_training_data(model; noise_length=min(length(synthetic_sounds), length(dev_voice_sounds)) ÷ 2, max_noise_level=0.01f0)

log("loaded $(size(training_data.input)[2]) training samples")
log("loaded $(size(synthetic_training_data.input)[2]) synthetic samples")
log("loaded $(size(denoising_training_data.input)[2]) denoising training samples")

training_data = (
    input = hcat(training_data.input, synthetic_training_data.input, denoising_training_data.input),
    target = hcat(training_data.target, synthetic_training_data.target, denoising_training_data.target),
    features = (
        f0s = hcat(training_data.features[:f0s], synthetic_training_data.features[:f0s], denoising_training_data.features[:f0s]),
        loudness = hcat(training_data.features[:loudness], synthetic_training_data.features[:loudness], denoising_training_data.features[:loudness]),
        mfccs = hcat(training_data.features[:mfccs], synthetic_training_data.features[:mfccs], denoising_training_data.features[:mfccs])
    )
)

test_data = nothing
test_sound = nothing

if local_rank == 0
    test_data_dir = "src/samples/LibriSpeech/test"
    test_data = get_training_data(model, vcat(load_sounds(test_data_dir)...))
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
        ys, _ = get_connected_trajectory(model, ps, st, (test_data.input, test_data.features))
        latent_xs = time_series_to_latent(model.encoder, ps.encoder, st.encoder, test_sound, model.stream_filter.cell.n)
        decoded = latent_to_time_series(model.decoder, ps.decoder, st.decoder, latent_xs)

        plt = plot(
            plot(eachcol(latent_xs)),
            plot([test_sound, decoded, ys], label=["test_sound" "decoded" "model output"]),
            size = (1200, 800),
            layout = @layout [a; b]
        )

        l = sum(abs2, test_sound .- ys)

        savefig(plt, "training_results/$(now())_$(l).png")
    catch e 
        @warn e
    finally
        @save "training_results/$(now())_hnode_vocoder_params.bson" ps
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
    if batch_i == batch_n || batch_i % BATCHES_BETWEEN_CHECKPOINTS == 0
        log_str = "$(now()): epoch $epoch, batch $batch_i/$batch_n"
        root_log(log_str)
        @save model_filename ps
    end

    return false
end

try
    ps, st = train_final_model(
        model, ps, st,
        training_data;
        batchsize = 32 ÷ total_workers,
        slices = 3,
        optimiser = Optimisers.ADAM(1e-4),
        epochs = 2,
        epoch_cb = epoch_cb,
        batch_cb = batch_cb,
        distributed_backend = backend,
        logger = log
    )
    break
catch e
    println(e)
end

end
