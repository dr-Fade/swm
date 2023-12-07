variable_power(α, β, c) = u -> begin
    if u < -c^(1/(β-1))
        # @info "u < -c^(1/(β-1))"
        -(β-1) * c^(β/(β-1)) / β - (-u)^β / β
    elseif u ≤ c^(1/(α-1))
        # @info "u ≤ -c^(1/(α-1))"
        c*u
    else
        # @info "else" c^(α/(α-1))
        (α-1) * c^(α/(α-1)) / α + u^α / α
    end
end

abs_square(params) = x -> x*abs(x)