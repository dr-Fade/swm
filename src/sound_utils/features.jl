
function f0_from_cepstrum(quefs, cepstrum; fmin=60.0, fmax=400.0)
    min_quef_index = findfirst(quefs .>= 1 / fmax)
    if min_quef_index === nothing 
        min_quef_index = 1
    end
    max_quef_index = findfirst(quefs .>= 1 / fmin)
    if max_quef_index === nothing
        max_quef_index = length(quefs)
    end
    
    valid_values = abs.(cepstrum[min_quef_index:max_quef_index])
    idx = findmax(valid_values)[2]
    return 1 / quefs[min_quef_index:max_quef_index][idx]
end
