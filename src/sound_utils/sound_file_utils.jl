sine_wave = (f0, A, ϕ) -> x -> Float32(A*sin(f0 * 2π * x + ϕ))

harmonic_signal = (f0, A, ϕ, harmonics, range) -> begin
    res = sum([
        sine_wave(harmonic*f0, 0.5f0 + 0.5f0rand(Random.default_rng(), Float32), ϕ).(range)
        for harmonic ∈ 1:harmonics
    ])
    res ./= max(maximum(res), 1)
    res .*= A
    res
end

function get_dirs_with_sound(root; file_extensions = [".flac", ".wav"])
    sound_dirs = []

    for (root, _, files) ∈ walkdir(root)
        if any(endswith(file, extension) for extension ∈ file_extensions, file ∈ files)
            push!(sound_dirs, root)
        end
    end

    return sound_dirs
end

# load and resample all sounds for training from directories
function load_sounds(target_dirs...; file_limit_per_dir = nothing, verbose = true, target_sample_rate = 16000, shuffle_files=false)
    fade_duration = (target_sample_rate * 0.1) |> Int # 100 millis
    w = DSP.Windows.hanning(2*fade_duration)
    fade_in = w[1:fade_duration]
    fade_out = w[fade_duration+1:end]

    result = vcat([
        begin
            verbose && println("loading files from $(target_dir)...")
            files = readdir(target_dir; join = true) |> (shuffle_files ? shuffle : identity)

            [
                begin
                    if endswith(f, "flac") || endswith(f, "wav")
                        sound, sample_rate = load(f)
                        if sample_rate != target_sample_rate
                            sound = resample(sound[:,1], target_sample_rate / sample_rate)
                        end
                        sound = sound[:,1]
                        sound[1:fade_duration] .*= fade_in
                        sound[end-fade_duration+1:end] .*= fade_out
                        Vector{Float32}(sound)
                    else
                        Vector{Float32}()
                    end
                end
                for f ∈ files[1:(isnothing(file_limit_per_dir) ? end : min(file_limit_per_dir, length(files)))]
            ]
        end
        for target_dir ∈ vcat(get_dirs_with_sound.(target_dirs)...)
    ]...)

    return result
end

function create_synthesized_data(N, f0_floor, f0_ceil, sample_rate; harmonics = 1, output_dir="synthesized_data", max_peak=1.0, f_step=10, noise = 0f0)
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    Δt = 1 / sample_rate
    T = N * Δt
    t = 0:Δt:T

    for f0 ∈ f0_floor:f_step:f0_ceil
        x = reduce(+, [sine_wave(f, 1, 0).(t) for f ∈ (1:harmonics) .* f0])
        x = noise*(2*rand(Float32, size(x)) .-1) .+ max_peak * (x ./ maximum(x))
        filename = if max_peak < 1.0
            joinpath(output_dir, "$(f0)_x$(harmonics)_$(max_peak)_$(sample_rate).wav")
        else
            joinpath(output_dir, "$(f0)_x$(harmonics)_$(sample_rate).wav")
        end
        wavwrite(x, filename; Fs = sample_rate)
    end
end

function create_synthetic_data(f0, A, ϕ, harmonics, info_file_name)
    x = harmonic_signal(f0, A, ϕ, harmonics)
    open(info_file_name,"a") do file
        println(file,"$(now()): epoch $epoch/$epochs, loss $(round(l; digits=2))")
    end
end