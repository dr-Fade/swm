using Lux, Random, LinearAlgebra

struct MergeLayer <: Lux.AbstractLuxLayer
    ins::Tuple
    out::Int
    connection
    σ::Function
    function MergeLayer(io, connection=+, σ=identity)
        ins, out = io
        new(ins, out, connection, σ)
    end

end

function (f::MergeLayer)(us, ps, st)
    Ws = ps[:Ws]
    bs = ps[:bs]

    W_keys = keys(Ws)
    b_keys = keys(bs)

    results = [
        f.σ.(Ws[W_key]*u .+ bs[b_key])
        for (u, W_key, b_key) ∈ zip(us, W_keys, b_keys)
    ]

    connected_results = reduce(f.connection, results)

    return connected_results, st
end

Lux.initialparameters(rng::AbstractRNG, f::MergeLayer) = (
    Ws = (; [Symbol("W_$i") => rand(rng, Float32, f.out, f.ins[i]) .- 0.5f0 for i ∈ 1:length(f.ins)]...),
    bs = (; [Symbol("b_$i") => zeros32(f.out) for i ∈ 1:length(f.ins)]...),
)

Lux.initialstates(rng::AbstractRNG, f::MergeLayer) = NamedTuple()
