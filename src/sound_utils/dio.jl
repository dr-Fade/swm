using LinearAlgebra, SignalAnalysis, DSP

const F0Estimate = @NamedTuple{value::Float32, confidence::Float32}

struct DIO
    f0_ceil::Float32
    f0_floor::Float32
    sample_rate::Integer
    decimated_frame_length::Int
    filters::Vector{FIRFilter}
    buffers::Vector{Vector{Float32}}
    τ_min::Int
    τ_max::Int
    relative_peak_threshold::Float32
    noise_floor::Float32

    DIO(f0_ceil, f0_floor, sample_rate; relative_peak_threshold = 0.25, noise_floor = 0.01) = begin
        decimated_frame_length = (3 * DOWNSAMPLED_RATE / f0_floor) |> round |> Int
        window_width = (sample_rate / f0_floor) |> floor |> Int
        window = DSP.Windows.tukey(window_width, 0.5) |> Vector{Float32} |> FIRWindow
        bands = Int.(exp2.(round(log2(f0_floor)):round(log2(f0_ceil)))[1:end])
        filters = [
            begin
                h = digitalfilter(DSP.Bandpass(f0_floor, band; fs=sample_rate), window) |> Vector{Float32}
                FIRFilter(h, DOWNSAMPLED_RATE // sample_rate)
            end
            for band ∈ bands
        ]
        new(
            f0_ceil,
            f0_floor,
            sample_rate,
            decimated_frame_length,
            filters,
            [zeros(Float32, decimated_frame_length) for i ∈ 1:length(bands)],
            DOWNSAMPLED_RATE / 3 / f0_ceil |> round |> Int,
            DOWNSAMPLED_RATE / 3 / f0_floor |> round |> Int,
            relative_peak_threshold,
            noise_floor
        )
    end
end

# decimate the signal and apply different filters to attenuate different harmonics
attenuate_harmonics(dio::DIO) = sample -> begin
    for ((i, f), buf) ∈ zip(enumerate(dio.filters), dio.buffers)
        shiftin!(buf, DSP.filt(f, sample) * log2(i+1f0))
    end
    dio.buffers
end

const Point2d = Tuple{Float32, Float32}

# find where line between p1 and p2 intersects the X axis
find_zero_intersection(p1::Tuple, p2::Tuple)::Float32 = begin
    x1, y1 = p1
    x2, y2 = p2
    -y1 * (x2 - x1) / (y2 - y1) + x1
end

# fit a parabola to 3 points and return its peak position
find_peak(p1::Tuple, p2::Tuple, p3::Tuple)::Point2d = begin
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    A = [x1^2 x1 1;
        x2^2 x2 1;
        x3^2 x3 1;]
    b = [y1, y2, y3]

    # find coefficients for basic parabola
    # y = ax² + bx + c
    a, b, c = A \ b

    x = -b / 2a
    y = -b^2 / 4a + c

    (x, y)
end

# util function that tries to find the first point that satisfies the current_detector.
# if a point is found, its index is returned with the range offset, e.g., if the point was
# searched for in 53:360 and was found at index 4, then the function will return 57.
apply_detector(dio::DIO, range::UnitRange, sample::Vector{Float32}, noise_floor::Float32, detector::Symbol, previous_peak=nothing)::Union{Int64, Nothing} = begin
    safe_range = range[dio.τ_min:min(dio.τ_max, length(range)-1)]
    res = if detector == :find_negative_peak
        findfirst(
            i ->
                sample[i] < -noise_floor &&
                # the point must be a peak
                sample[i-1] > sample[i] < sample[i+1] &&
                # the peak must be at least α times the previous positive peak
                (isnothing(previous_peak) || abs(sample[i]) > dio.relative_peak_threshold*previous_peak),
                safe_range
        )
    elseif detector == :find_negative_zero_crossing
        findfirst(
            i -> sample[i-1] > 0.0 && sample[i] < 0.0,
            safe_range
        )
    elseif detector == :find_positive_peak
        findfirst(
            i ->
                sample[i] > noise_floor &&
                # the point must be a peak
                sample[i-1] < sample[i] > sample[i+1] &&
                # the peak must be at least α times the previous negative peak
                (isnothing(previous_peak) || sample[i] > abs(dio.relative_peak_threshold*previous_peak)),
            safe_range
        )
    else
        findfirst(
            i -> sample[i-1] < 0.0 && sample[i] > 0.0,
            safe_range
        )
    end
    if !isnothing(res)
        offset = (dio.τ_min - 1) + (range[begin] - 1)
        res + offset
    else
        nothing
    end
end

detectors = [:find_positive_zero_crossing; :find_positive_peak; :find_negative_zero_crossing; :find_negative_peak]

get_next_detector(detector) = 
    detectors[findfirst(x -> x == detector, detectors) % 4 + 1]

# a single cycle goes through 4 detectors; we need 2 points for each detector to determine periods, i.e. 8 detections
full_period = 2length(detectors)

# apply the current detector to a range and continue the search with the next detector in the chain until depth is reached
apply_detectors_chain(dio::DIO, range::UnitRange{Int64}, sample::Vector{Float32}, noise_floor::Float32, depth::Int = 1, previous_peak = nothing; detector = nothing)::Union{Vector{Union{Float32, Nothing}}, Nothing} = begin
    res = nothing

    # attempt find initial detector
    if isnothing(detector)
        # only search in the vicinity of the largest/lowest peak
        init_range = sample[1:end-Int(2*DOWNSAMPLED_RATE÷dio.f0_floor)]
        start_index = max(min(findmax(init_range)[2], findmin(init_range)[2]) - Int(DOWNSAMPLED_RATE÷dio.f0_floor÷4), 1)
        end_index = start_index + 2dio.τ_max
        for d ∈ detectors
            for i ∈ start_index:dio.τ_min:end_index
                r = apply_detector(dio, i:range[end], sample, noise_floor, d)
                if !isnothing(r) && (isnothing(res) || r < res)
                    detector = d
                    res = r
                    break
                end
            end
        end
    else
        res = apply_detector(dio, range, sample, noise_floor, detector, previous_peak)
    end

    if isnothing(res)
        return nothing
    end

    # the result of the next peak detections depends on the magnitude of the previous peak, so we'll have to memorize it
    previous_peak = if detector == :find_positive_peak || detector == :find_negative_peak
        # it's unlikely that the sample would land exactly on the wave's peak, especially after decimation
        # to improve accuracy, we'll fit a parabola real quick and use its max/min as the peak's time offset and amplitude
        interp_peak = find_peak(
            (res-1, sample[res-1]),
            (res, sample[res]),
            (res+1, sample[res+1])
        )
        res = interp_peak[1]
        interp_peak[2]
    else
        # build a line between the points where the sample crosses zero and calculate the exact point where the intersection
        # happens
        res = find_zero_intersection((res, sample[res]), (res-1, sample[res-1]))
        previous_peak
    end

    if depth > 1
        # if the current detector was looking for zero crossings, its result will lie somewhere between the actual indeces.
        # rounding the result will ensure the construction of a valid range.
        new_range_start = (round(res) |> Int)
        [res; apply_detectors_chain(dio, new_range_start:range[end], sample, noise_floor, depth-1, previous_peak; detector=get_next_detector(detector))]
    else
        [res]
    end
end

detect_pitch(dio::DIO) = sample::Vector{Float32} -> begin
    rms_sample = rms(sample)
    if rms_sample < dio.noise_floor
        return (value=0f0, confidence=0f0)
    end

    noise_floor = max(rms_sample / 4, dio.noise_floor)

    f0, Λ = 0f0, 0f0

    range = 1:length(sample)
    detected_points = apply_detectors_chain(dio, range, sample, noise_floor, full_period)
    if !isnothing(detected_points) && !any(isnothing, detected_points)
        lpz, lpp, lnz, lnp = (detected_points[5:8] - detected_points[1:4]) ./ DOWNSAMPLED_RATE
        T = (lpz + lpp + lnz + lnp) / 4
        Λ = 1 - min((abs(lpz-T) + abs(lpp-T) + abs(lnz-T) + abs(lnp-T)) / 4T, 1)
        f0 = 1 / T
    end

    (value=f0, confidence=Λ)
end

# find the f0 estimation with the highest likelihood value λ and lowest distance from the previous estimate
select_candidate(previous_estimate::F0Estimate = (value=0f0, confidence=0f0)) = estimates::Vector{F0Estimate} -> begin
    if isempty(estimates)
        return (value=0f0, confidence=0f0)
    end

    _, i = findmax(e -> e.confidence, estimates)
    candidate = estimates[i]

    if candidate.confidence < previous_estimate.confidence
        _, j = findmin(e -> abs(e.value - previous_estimate.value), estimates)
        candidate = (value = estimates[j].value, confidence = candidate.confidence)
    end
    
    return candidate
end

const DOWNSAMPLED_RATE::Int = 4000

# the algorithm is lifted straight from the article:
# https://www.isca-speech.org/archive/pdfs/interspeech_2016/daido16_interspeech.pdf
(dio::DIO)(sample::Vector{Float32}; previous_estimate::F0Estimate = (value=0f0, confidence=0f0)) = sample |>
    attenuate_harmonics(dio) .|>
    detect_pitch(dio) |>
    select_candidate(previous_estimate)

select_candidates(estimates::Vector{Vector{F0Estimate}}) = begin
    N = estimates[1] |> length
    prev_estimate = (0.0, 0.0)
    contour = Vector{Tuple{Float32, Float32}}(undef, N)
    for i ∈ 1:N
        candidates = [estimates[j][i] for j ∈ eachindex(estimates)]
        if i > 1
            prev_estimate = contour[i-1]
        end
        contour[i] = candidates |> select_candidate(prev_estimate)
    end
    contour
end

# return an estimated f0 curve and time axis at which the values were detected
dio_contour(dio::DIO, sound::Vector{Float32}; hop::Float32 = 1f0)::Vector{F0Estimate} = begin
    N = length(sound)
    n = hop * dio.sample_rate / 1000f0 |> round |> Int
    res = [(value=0f0, confidence=0f0)]
    for frame ∈ eachcol(reshape(sound[1:end-N%n], n, N÷n))
        estimate = dio(Vector(frame); previous_estimate = res[end])
        push!(res, estimate)
    end
    return res
end
    
