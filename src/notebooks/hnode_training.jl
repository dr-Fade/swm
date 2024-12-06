include("../nn/hnode.jl")
include("../nn/training_utils.jl")
include("../nn/activation_functions.jl")
include("../nn/conv_1d.jl")
include("../sound_utils/sound_file_utils.jl")

using DSP, FFTW, BSON, Plots, Dates

const f0_floor = 60f0
const f0_ceil = 600f0
const sample_rate = 16000f0
const actual_dt = 1f0 / sample_rate

const input_length = Int(sample_rate ÷ f0_floor)
const embedding_dimension = 10;

const model_dt = 1000f0*actual_dt
const model_T = input_length * model_dt;

function create_model()
    encoder_input = Int(sample_rate ÷ f0_floor)
    kernel_n = Int(3 * sample_rate ÷ f0_ceil)
    encoder_conv_layer, encoder_conv_output_n = conv_1d(encoder_input, kernel_n)
    encoder = Chain(
        x -> x[1:encoder_input, :],
        encoder_conv_layer,
        SkipConnection(Dense(encoder_conv_output_n, encoder_conv_output_n, tanh), +),
        SkipConnection(Dense(encoder_conv_output_n, encoder_conv_output_n, tanh), +),
        Dense(encoder_conv_output_n, embedding_dimension)
    )

    decoder = Dense(embedding_dimension, 1)

    ode = Parallel(+,
        Dense(embedding_dimension, embedding_dimension),
        Dense(embedding_dimension, embedding_dimension, sqrt_activation)
    )
    ode_params_n = Lux.initialparameters(Random.default_rng(), ode) |> ComponentArray |> length

    kernel_n = Int(3 * sample_rate ÷ f0_ceil)
    control_conv_layer, control_conv_output_n = conv_1d(input_length, kernel_n)
    control = Chain(
        control_conv_layer,
        ReshapeLayer((1, control_conv_output_n)),
        Recurrence(LSTMCell(1=>128)),
        Dense(128, 512, tanh),
        repeat([SkipConnection(Dense(512, 512, tanh), +)], 10),
        Dense(512, ode_params_n)
    )

    hnode = hNODE(encoder, decoder, control, ode)
    ps, st = Lux.setup(Random.default_rng(), hnode)
    if isfile("stats/hnode.bson")
        BSON.@load "stats/hnode.bson" model_stats
        ps, st = model_stats.ps, model_stats.st
        @info "Loading parameters from stats/hnode.bson"
    else
        @info "Creating new parameters"
    end 
    return hnode, ps, (st..., Δt = model_dt, T = input_length * model_dt)
end


create_data(input_length, max_harmonics; amp = 1f0) = begin
    f0s = f0_floor:f0_ceil
    timestamps = actual_dt .* (0:input_length-1)

    training_data = Matrix{Float32}(undef, input_length, max_harmonics*length(f0s))
    for (i, (harmonics, f0)) ∈ enumerate(Iterators.product(1:max_harmonics, f0s))
        training_data[:,i] = harmonic_signal(f0, 1f0, 10*rand(Float32), harmonics, timestamps)
    end

    amp*training_data
end

batch_cb(model_name; reset_stats::Bool=false, dir_name = "stats", test_data = create_data(input_length, 3)) = (loss_function, model, ps, st, batch_i, batch_N, epoch) -> begin
    cb(model_name; reset_stats=reset_stats, dir_name = dir_name, test_data = test_data)(loss_function, epoch, 8, model, ps, st)
end

cb(model_name; reset_stats::Bool=false, dir_name = "stats", test_data = create_data(input_length, 3)) = (loss_function, epoch, epochs, model, ps, st) -> begin
    if any(isnan, ComponentArray(ps))
        error("NaN detected!")
    end

    if !isdir(dir_name)
        mkdir(dir_name)
    end

    model_stats_filename = "$dir_name/$model_name.bson"
    bkp_model_stats_filename = "$dir_name/$(model_name)_$(now()).bson"
    stats_plot_filename = "$dir_name/$model_name.png"

    if reset_stats && epoch == 1
        for f ∈ [model_stats_filename, stats_plot_filename]
            if isfile(f)
                rm(f)
            end
        end
    end

    model_stats = (loss = Vector{Float32}(), ps = NamedTuple(), st = NamedTuple())
    if isfile(model_stats_filename)
        BSON.@load model_stats_filename model_stats
    end

    model_stats = (model_stats..., ps = ps, st = st)
    # try
    #     push!(model_stats.loss, loss_function(model, ps, st, (test_data,)))
    #     plt = plot(model_stats.loss, title = "Test data loss", size = (1600, 800))
    #     savefig(plt, stats_plot_filename)
    # catch e
    #     @warn e
    # end

    BSON.@save model_stats_filename model_stats
    if epoch == epochs
        BSON.@save bkp_model_stats_filename model_stats
    end

    return false
end

function train_trajectories(hnode, ps, st; max_harmonics = 3, epochs = 1, optimiser = Optimisers.Adam(0.003), n = input_length, reset_stats = false, control_n = input_length)
    training_loss(model, ps, st, data) = begin
        xs, ys = data
        n = size(ys)[1]
        batches = size(xs)[end]

        u0s, encoder_st = model.encoder(xs, ps.encoder, st.encoder)
        controls, control_st = model.control(xs[1:control_n,:], ps.control, st.control)

        if any(isnan, controls)
            error("controls ara NaN: $(controls)")
        end

        pred_ys, model_st = model((xs, (u0s, controls)), ps, (st..., T = model_dt * n))
        pred_ys = pred_ys[1][:, 1, :]
    
        loss = 1000f0 * sum(abs2, pred_ys .- ys) / n / batches
        if isnan(loss)
            error("loss is NaN: $(pred_ys[end,:])")
        end

        params_restrictions = sum(abs, ComponentArray(ps)) / length(ComponentArray(ps))
        regularization = sum(abs, controls) / length(controls)

        return loss + params_restrictions + regularization, (model_st..., control = control_st, encoder = encoder_st), (nothing,)
    end

    if isnothing(ps)
        ps, st = Lux.setup(Random.default_rng(), hnode)
        @info "Creating new parameters"
    end

    training_data = create_data(input_length, max_harmonics; amp = 1f0)
    return train(
        hnode,
        DataLoader((training_data + 0.05f0.*rand32(size(training_data)...), training_data[1:n, :]); batchsize = 64, shuffle = true),
        training_loss;
        ps = ps,
        st = st,
        batch_cb = batch_cb("hnode"; reset_stats = reset_stats && epoch == 1),
        epochs = epochs,
        optimiser = optimiser
    )
end

# let the model integrate for a significant time and compare the spectral characteristics
# of the attractor with the input signal's.
#
# this will train the control and the decoder and the control models
function train_attractors(hnode, ps, st; max_harmonics = 3, epochs = 1, optimiser = Optimisers.Adam(0.003), delay = model_T, reset_stats = false)
    window = hanning(input_length)
    training_loss(model, ps, st, data) = begin
        target_signal, = data
        n, batches = size(target_signal)
        noise = 0.001f0 * rand32(n, batches)
        input_signal = target_signal + noise

        u0s = rand32(embedding_dimension, batches) .- 0.5f0
        T = delay + n * model_dt
        controls, control_st = model.control(input_signal, ps.control, st.control)

        pred_signal, model_st = model((target_signal, (u0s, controls)), ps, (st..., T = T))

        attractor = pred_signal[1][end-n+1:end, 1, :]

        target_spectrum = log10.(abs.(fft(window .* target_signal, 1)))
        actual_spectrum = log10.(abs.(fft(window .* attractor, 1)))

        loss = sum(abs2, target_spectrum .- actual_spectrum)

        return loss / batches, (model_st..., control = control_st), (nothing,)
    end

    if isnothing(ps)
        ps, st = Lux.setup(Random.default_rng(), hnode)
        @info "Creating new parameters"
    end

    training_data = create_data(input_length, max_harmonics)
    return train(
        hnode,
        DataLoader((training_data,); batchsize = 128, shuffle = true),
        training_loss;
        ps = ps,
        st = st,
        epoch_cb = cb("hnode"; reset_stats = reset_stats && epoch == 1),
        epochs = epochs,
        optimiser = optimiser
    )
end
