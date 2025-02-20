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
dev_voice_files = shuffle(get_all_sound_files("src/samples/vowels/"))[1:2:end][(local_rank+1):total_workers:end]
synthetic_files = shuffle(get_all_sound_files("src/samples/synthesized_sounds"))[1:5:end][(local_rank+1):total_workers:end]
files_n = length(synthetic_files) + length(dev_voice_files)

log("picked $(files_n) files")

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
denoising_training_data = get_denoising_training_data(model; noise_length=min(length(synthetic_sounds), length(dev_voice_sounds)) ÷ 2, max_noise_level=0.02f0)

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

       # try
       #     target_s = 0.793125
       #     start_sample = target_s * FEATURE_EXTRACTION_SAMPLE_RATE |> floor |> Int
       #     samples_n = 512
       #     start_frame = start_sample ÷ model.n
       #     frames_n = samples_n ÷ model.n

       #     ys, _ = get_connected_trajectory(
       #         model, ps, st,
       #         (test_data.input[:,start_frame:start_frame+frames_n-1], Tuple(v[:, start_frame:start_frame+frames_n-1] for v in values(test_data.features)))
       #     )

       #     target = test_sound[(start_frame-1)*model.n+1:(start_frame+frames_n-1)*model.n]

       #     plt = plot(target, size=(1200,600), label="target")
       #     plot!(plt, ys, label="generated")
       #     savefig(plt, "training_results/trajectories_demo_$(now()).png")

       #     end_sample = start_sample + samples_n
       #     latent_xs = time_series_to_latent(model.encoder, ps.encoder, st.encoder, test_sound[start_sample:(end_sample+size(test_data.input[:,1])[1])], model.stream_filter.cell.n)
       #     # decoded = latent_to_time_series(model.decoder, ps.decoder, st.decoder, latent_xs);

       #     # plt = plot(test_sound[start_sample:end_sample], size=(1200,600), label = "target")
       #     # plot!(plt, decoded, label = "decoded")
       #     # savefig(plt, "training_results/autoencoder_demo_$(now()).png")

       #     plt = plot(
       #         [plot(latent_xs[:,i], latent_xs[:,i+1], latent_xs[:,i+3]) for i ∈ 1:2:LATENT_DIMS-3]...,
       #         size=(1200,600)
       #     )
       #     savefig(plt, "training_results/latent_demo_$(now()).png")
       #     
       #     control = model.control(
       #         (test_data.features.f0s, test_data.features.loudness, test_data.features.mfccs),
       #         ps.control,
       #         st.control
       #     )[1]
       #     linear_eigens = hcat([real.(eigvals(ComponentArray(x, model.ode_axes).linear_block.weight)) for x in eachcol(control)]...)
       #     nonlinear_eigens = hcat([real.(eigvals(ComponentArray(x, model.ode_axes).nonlinear_block.weight)) for x in eachcol(control)]...)
       #     plt = plot(
       #         plot(eachrow(linear_eigens)),
       #         plot(eachrow(nonlinear_eigens)),
       #         size=(1600,800),
       #         layout=@layout [a;b]
       #     )
       #     savefig(plt, "training_results/eigens_demo_$(now()).png")
       # catch e
       #     @warn e
       # end

        log_str = "$(now()): epoch $epoch, batch $batch_i/$batch_n"
        root_log(log_str)
        @save model_filename ps
        @save model_state_filename st
    end

    return false
end

batchsize = 256 ÷ total_workers
optimizer_step = total_workers * batchsize * 1e-6
slices = 2
root_log("model.n = $(model.n), batchsize = $(batchsize), slices = $(slices), optimizer step = $(optimizer_step)")

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
