using FFTW

q1 = -0.09
q0 = 1 - 2q1

lq(q0::Float64, q1::Float64, f0::Float64, τ::Float64)::Float64 =  q0 + 2q1 * cos(2π*τ*f0)
ls(f0::Float64, τ::Float64)::Float64 =
    if τ == 0
        1.0
    else
        sin(f0 * π * τ)/(f0 * π * τ)
    end

struct CheapTrick
    f0_ceil::Float64
    f0_floor::Float64
    sample_rate::Float64
    fft_size::Int
    default_frame_size::Int
    default_window::Vector{Float64}
    default_window_sum::Float64
    padded_sample::Vector{Float64}
    spectrum_zeros_safeguard::Vector{Float64}
    CheapTrick(f0_ceil, f0_floor, sample_rate) = begin
        frame_size = 3 * (sample_rate÷f0_ceil |> Int)
        fft_size = nextfastfft(2 ^ ceil(log2(3 * sample_rate / f0_floor + 1)-1) |> Int)
        window = hanning(frame_size)
        spectrum_zeros_safeguard = rand(fft_size)
        spectrum_zeros_safeguard ./= min((spectrum_zeros_safeguard ./ eps())...)
        new(
            f0_ceil,
            f0_floor,
            sample_rate,
            fft_size,
            frame_size,
            window,
            sum(window),
            zeros(2fft_size),
            spectrum_zeros_safeguard
        )
    end
end

# The algorithm was lifted directly form the article - https://www.sciencedirect.com/science/article/pii/S0167639314000697#s0015
cheaptrick_jl(ct::CheapTrick, f0::Float64, sample::Vector{Float64})::Vector{Float64} = begin
    # DIO returns 0 if it fails to detect pitch, use ceil f0 as a fallback because higher f0 == wider smoothing windows
    # the frequency can be too low to fit into the sample, so we'll have to force the frame size
    # to not fit.
    f0 = if f0 < ct.f0_floor ct.f0_ceil else f0 end
    frame_size = min(3 * (ct.sample_rate ÷ f0 |> Int), length(sample))

    # First step: F0-adaptive windowing
    # if possible, grab 3 times as many samples as contained within the full period of f0
    # and pad the sample with zeros to achieve fft of correct length
    window, fft_scaling = if frame_size == ct.default_frame_size
        ct.default_window, ct.default_window_sum
    else
        w = hanning(frame_size)
        w, sum(w)
    end
    ct.padded_sample[:] .= 0.0
    ct.padded_sample[1:frame_size] = sample[1:frame_size] .* window

    # Second step: frequency domain smoothing of the power spectrum
    spectrum = (abs2.(fft(ct.padded_sample)[1:ct.fft_size]) / fft_scaling) + ct.spectrum_zeros_safeguard

    # the width of the window should be 2/3 of the f0 but the spectrum is discretized into bins
    # so the width should be measured in number of bins instead of raw Hzs
    df = ct.sample_rate / 2ct.fft_size |> round |> Int
    f0_index = f0 ÷ df |> round |> Int
    smoothing_window_size = f0_index |> round |> Int
    smoothing_window = ones(smoothing_window_size) ./ smoothing_window_size

    # filter introduces some lag, we compensate for it by padding the input and then shifting the output
    smoothed_spectrum = filtfilt(smoothing_window, spectrum)

    # Third step: liftering in the quefrency domain
    # the cepstrum is calculated with regular fft instead of a reverse one because it avoids the fft
    # mirroring issues
    cepstrum = fft(smoothed_spectrum[1:ct.fft_size] .|> log)
    τ0 = 1 / f0
    for i ∈ 1:ct.fft_size÷2
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
    smoothed_spectrum[1:ct.fft_size] = ifft(cepstrum) .|> real .|> exp
    map(x -> isnan(x) ? 0.0 : x, smoothed_spectrum)
end
