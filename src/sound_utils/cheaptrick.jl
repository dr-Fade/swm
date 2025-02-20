using FFTW, Statistics

q1 = -0.09f0
q0 = 1f0 - 2q1

lq(q0::Float32, q1::Float32, f0::Float32, τ::Float32)::Float32 =  q0 + 2q1 * cos(2π*τ*f0)
ls(f0::Float32, τ::Float32)::Float32 =
    if τ == 0
        1.0
    else
        sin(f0 * π * τ)/(f0 * π * τ)
    end

struct CheapTrick
    f0_ceil::Float32
    f0_floor::Float32
    sample_rate::Float32
    fft_size::Int
    max_frame_size::Int
    padded_sample::Vector{Float32}
    CheapTrick(f0_ceil, f0_floor, sample_rate) = begin
        max_frame_size = nextfastfft(2 ^ ceil(log2(3 * sample_rate / f0_floor + 1)) |> Int)
        fft_size = length(rfft(zeros32(max_frame_size)))
        new(
            f0_ceil,
            f0_floor,
            sample_rate,
            fft_size,
            max_frame_size,
            zeros(max_frame_size)
        )
    end
end

# The algorithm was lifted directly form the article - https://www.sciencedirect.com/science/article/pii/S0167639314000697#s0015
(ct::CheapTrick)(f0::Float32, sample::Vector{Float32})::Vector{Float32} = begin
    # DIO returns 0 if it fails to detect pitch, use ceil f0 as a fallback because higher f0 == wider smoothing windows
    # the frequency can be too low to fit into the sample, so we'll have to force the frame size
    # to not fit.
    f0 = min(max(f0, ct.f0_floor), ct.f0_ceil)
    frame_size = min(3 * (ct.sample_rate ÷ f0 |> Int), length(sample))

    # First step: F0-adaptive windowing
    # if possible, grab 3 times as many samples as contained within the full period of f0
    # and pad the sample with zeros to achieve fft of correct length
    window = hanning(frame_size)
    ct.padded_sample[:] .= 0.0f0
    ct.padded_sample[1:frame_size] = sample[1:frame_size] .* window

    # Second step: frequency domain smoothing of the power spectrum
    spectrum = log10.(abs.(fft(ct.padded_sample)) .+ eps(Float32))

    # the width of the window should be 2/3 of the f0 but the spectrum is discretized into bins
    # so the width should be measured in number of bins instead of raw Hzs
    df = ct.sample_rate / ct.fft_size |> round |> Int
    smoothing_window_size = 2 * f0 ÷ df |> round |> Int
    smoothing_window = ones32(smoothing_window_size) ./ smoothing_window_size

    # filter introduces some lag, we compensate for it by padding the input and then shifting the output
    smoothed_spectrum = filtfilt(smoothing_window, spectrum)[1:ct.fft_size]

    # Third step: liftering in the quefrency domain
    # the cepstrum is calculated with regular fft instead of a reverse one because it avoids the fft
    # mirroring issues
    cepstrum = fft(smoothed_spectrum)
    τ0 = 1 / f0
    for i ∈ length(cepstrum)÷2
        τ = (i-1) / ct.sample_rate
        if τ < τ0
            lifter = ls(f0, τ) * lq(q0, q1, f0, τ)
            cepstrum[i] *= lifter
            cepstrum[end-i+1]*= lifter
        else
            cepstrum[i] = 0.0
            cepstrum[end-i+1] = 0.0
        end
    end
    map(x -> isnan(x) ? floatmin(Float32) : x, ifft(cepstrum) .|> real)
end
