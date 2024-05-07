variable_power(α, β, c) = u -> begin
    if u < -c^(1/(β-1))
        -(β-1) * c^(β/(β-1)) / β - (-u)^β / β
    elseif u ≤ c^(1/(α-1))
        c*u
    else
        (α-1) * c^(α/(α-1)) / α + u^α / α
    end
end

# α == β == c == 0.5
sqrt_activation(u) = begin
    if u < -4
        2 - 2*sqrt(-u)
    elseif u ≤ 4
        u / 2
    else
        -2 + 2sqrt(u)
    end
end

abs_square(params) = x -> x*abs(x)