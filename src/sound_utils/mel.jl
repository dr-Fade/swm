hz_to_mel(hz) = 2595 * log10(1 + hz / 700)
mel_to_hz(mel) = 700 * (10 ^ (mel / 2595) - 1)

function fft_to_mel_bins(freqs::Vector{Float64}; k = 30)
    N = length(freqs)
    filters = zeros(k, N)
    minmel = hz_to_mel(freqs[begin])
    maxmel = hz_to_mel(freqs[end])
    binfreqs = mel_to_hz.(minmel .+ collect(0:(k+1)) / (k+1) * (maxmel - minmel))

    for i in 1:k
        fs = binfreqs[i .+ (0:2)]
        # scale by width
        fs = fs[2] .+ (fs .- fs[2])
        # lower and upper slopes for all bins
        loslope = (freqs .- fs[1]) / (fs[2] - fs[1])
        hislope = (fs[3] .- freqs) / (fs[3] - fs[2])
        # then intersect them with each other and zero
        filters[i,:] = max.(0, min.(loslope, hislope))
    end
    return filters
end