variable_power(powers) = x -> begin
    α, β = abs.(powers)
    if x < 0
        -(-x)^α
    else
        x^β
    end
end

abs_square(params) = x -> x*abs(x)