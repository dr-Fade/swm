using DSP, Lux

struct StreamFilter <: Lux.AbstractRecurrentCell{false, false}
    pregain::Float32
    gain::Float32
    filter::FIRFilter
    n::Integer

    function StreamFilter(n::Int, filter::FIRFilter; gain=1f0, pregain=1f0)
        return new(pregain, gain, filter, n)
    end

    function StreamFilter(n::Int, sample_rate::Int;
        target_sample_rate::Int=sample_rate,
        f0_floor::Float32=20f0,
        f0_ceil::Float32=1000f0,
        filter_length::Int=Int(sample_rate÷f0_floor),
        window=hanning,
        pregain=1f0,
        gain=1f0
    )
        h = digitalfilter(DSP.Bandpass(f0_floor, f0_ceil; fs=sample_rate), FIRWindow(window(filter_length)))
        fir_filter = FIRFilter(h, target_sample_rate // sample_rate)
        return new(pregain, gain, fir_filter, n)
    end
end

function (sf::StreamFilter)(xs::AbstractMatrix, ps, st::NamedTuple)
    reset!(sf.filter)

    m = size(xs)[2]
    buffers = zeros(Float32, sf.n, m)
    filters = [deepcopy(sf.filter) for i ∈ 1:m]

    return sf((xs, (buffers, filters)), ps, st)
end

function (sf::StreamFilter)((xs, (buffers, filters))::Tuple{<:AbstractMatrix, Tuple}, ps, st::NamedTuple)
    for (x, filter, buffer) ∈ zip(eachcol(xs), filters, eachcol(buffers))
        y = sf.gain * DSP.filt(filter, sf.pregain * x) |> Vector{eltype(xs)}
        shiftin!(buffer, y)
    end
    return (copy(buffers), (buffers, filters)), st
end
