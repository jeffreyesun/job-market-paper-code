
function get_H_S(q, params, H_S_prior=0)
    (;Π, elasticity, δ_dep) = params
    q = max.(q, 0)
    H_S = @. Π*q^elasticity
    H_S = max.(H_S, H_S_prior*(1-δ_dep))
    assert_all_finite(H_S)
    return H_S
end
get_H_S!(params, q, H_S_prior=0) = params.spatial.H_S .= get_H_S(q, params, H_S_prior)
