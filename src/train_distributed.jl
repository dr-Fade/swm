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

model = hNODEVocoder(FEATURE_EXTRACTION_SAMPLE_RATE; output_n = 5*Integer(FEATURE_EXTRACTION_SAMPLE_RATE ÷ 1000), speaker = false)

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
    vcat(
        load_sounds(synthetic_data; target_sample_rate=FEATURE_EXTRACTION_SAMPLE_RATE, verbose = local_rank==0, shuffle_files = true),
        load_sounds(dev_voice_sounds; target_sample_rate=FEATURE_EXTRACTION_SAMPLE_RATE, file_limit_per_dir = 4, verbose = local_rank==0, shuffle_files = true)
    ),
    backend.comm
)

root_log("loaded $(length(sounds)) files in total")

# filter out the files based on the local rank of each worker
sounds = sounds[local_rank+1:total_workers:end]

log("picked $(length(sounds)) files")

concatenated_sounds = vcat([vcat(zeros(Float32, 1000), s, zeros(Float32, 1000)) for s in sounds]...)
N = MPI.Reduce(length(concatenated_sounds), min, backend.comm)
N = MPI.bcast(N, backend.comm)

concatenated_sounds = concatenated_sounds[1:N]

training_data = get_training_data(model, concatenated_sounds, FEATURE_EXTRACTION_SAMPLE_RATE)
denoising_training_data = get_denoising_training_data(model; noise_length=max(length(concatenated_sounds)÷20, 1024))

log("loaded $(size(training_data[1])[2]) training samples")
log("loaded $(size(denoising_training_data[1])[2]) denoising training samples")

training_data = (hcat(training_data[1], denoising_training_data[1]), hcat(training_data[2], denoising_training_data[2]))

test_data = nothing
test_sound = nothing

if local_rank == 0
    test_data = get_training_data(model, vcat(load_sounds(test_data_dir; lowpass=1000f0)...), FEATURE_EXTRACTION_SAMPLE_RATE)
    test_sound = vcat(eachcol(test_data[1][1:model.output_n,:])...)
    log("loaded $(size(test_data[1])[2]) test samples")
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
        latent_xs = time_series_to_latent(model.encoder, ps.encoder, st.encoder, test_sound, model.input_n)
        decoded = latent_to_time_series(model.decoder, ps.decoder, st.decoder, latent_xs)
        ys, _ = get_connected_trajectory(model, ps, st, test_data)

        plt = plot(
            plot(eachcol(latent_xs)),
            plot([test_sound, decoded, ys], label=["test_sound" "decoded" "model output"]),
            heatmap(ps.encoder.layer_4.weight'),
            size = (1200, 800),
            layout = @layout [[a; b] c]
        )

        l = sum(abs2, test_sound .- ys)

        savefig(plt, "training_results/$(now())_$(l).png")
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
    batchsize = 16,
    slices = 10,
    optimiser = Optimisers.ADAM(1e-4),
    epochs = 1,
    epoch_cb = epoch_cb,
    batch_cb = batch_cb,
    distributed_backend = backend
)
