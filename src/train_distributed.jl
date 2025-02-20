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

model = hNODEVocoder(FEATURE_EXTRACTION_SAMPLE_RATE; n = Integer(FEATURE_EXTRACTION_SAMPLE_RATE ÷ F0_CEIL) ÷ 2)

while true

rng = Random.default_rng()
model_filename = ARGS[1]
model_state_filename = "src/notebooks/hnode_vocoder_state.bson"
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
# dev_voice_files = repeat(get_all_sound_files("src/samples/LibriSpeech/demo"), 20)
# dev_voice_files = get_all_sound_files("src/samples/LibriSpeech/dev-clean/1272")[1:20][(local_rank+1):total_workers:end] |> shuffle
dev_voice_files = get_all_sound_files("src/samples/vowels/")[(local_rank+1):total_workers:end]
synthetic_files = get_all_sound_files("src/samples/synthesized_sounds")[(local_rank+1):total_workers:end]
all_files = vcat(dev_voice_files, synthesized_sounds) |> shuffle
log("picked $(length(all_files)) files")

batchsize = 512 ÷ total_workers
optimizer_step = total_workers * batchsize * 1e-6
slices = 2
root_log("model.n = $(model.n), batchsize = $(batchsize), slices = $(slices), optimizer step = $(optimizer_step)")

for file in all_files

    # pad the sound to ensure all workers finish at the same time
    sound = get_sound(file)
    N = MPI.Reduce(length(sound), min, backend.comm)
    N = MPI.bcast(N, backend.comm)
    sound = vcat(sound, zeros32(length(sound) - N))

    training_data = get_training_data(model, sound)
    
    log("loaded $(size(training_data.input)[2]) training samples")

    test_data = nothing
    test_sound = nothing

    if local_rank == 0
        test_data_dir = "src/samples/LibriSpeech/demo"
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
            st = Lux.testmode(st)
            ys, _ = get_connected_trajectory(model, ps, st, test_data)
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
            st = Lux.testmode(st)
            log_str = "$(now()): epoch $epoch, batch $batch_i/$batch_n"
            root_log(log_str)
            @save model_filename ps
            @save model_state_filename st
        end

        return false
    end

    try
        ps, st = train_final_model(
            model, ps, Lux.trainmode(st),
            training_data;
            batchsize = batchsize,
            slices = slices,
            optimiser = Optimisers.ADAM(optimizer_step),
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
end