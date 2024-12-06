include("hnode_training.jl")

function main()
    ns = repeat(26:266, 2) |> shuffle

    for n ∈ ns
        while true
            try
                hnode, ps, st = create_model()
                @info "training on n=$n"
                ps, st = train_trajectories(hnode, ps, st; epochs = 8, n = n, optimiser = Optimisers.Adam(0.003), max_harmonics=1)
                # ps, st = train_attractors(hnode, ps, st; epochs = 8, delay=24f0)
                break
            catch e
                @error e
                if typeof(e) == InterruptException
                    throw(e)
                end
            end
        end
    end
end

main()
