using LinearAlgebra

function to_spherical(x::Vector{T})::Vector{T} where T<: Real
    n = length(x)
    y = zero(x)
    @inbounds begin
        y[1] = norm(x)  # r
        for i in 1:n-2
            y[i+1] = acos(x[i] / norm(x[i:n]))
        end
        if x[n] >= 0
            y[n] = acos(x[n-1] / norm(x[n-1:n]))
        else
            y[n] = 2π - acos(x[n-1] / norm(x[n-1:n]))
        end
    end
    return y
end

function to_cartesian(y::Vector{T})::Vector{T} where T<: Real
    r = y[1]
    ϕ = y[2:end]
    return to_cartesian(r, ϕ)
end

function to_cartesian(r::T, ϕ::Vector{T})::Vector{T} where T<: Real
    n = length(ϕ) + 1
    x = zeros(T, n)
    @inbounds begin
        x[1] = r * cos(ϕ[1])
        for i in 2:n-1
            x[i] = r * prod(sin.(ϕ[1:i-1])) * cos(ϕ[i])
        end
        x[n] = r * prod(sin.(ϕ[1:n-1]))
    end
    return x
end

function construct_expriment_matrix(η::Matrix{T}, M̄::Array{T, 3})::Matrix{T} where T <: Real
    nγ = size(η, 2)
    Ē = Matrix{T}(undef, 6 * nγ, nα)
    @inbounds begin
        for α in 1:nα
            for γ in 1:nγ
                Ē[(6 * γ - 5):(6 * γ), α] = M̄[:,:,α] * η[:,γ]
            end
        end
    end
    return Ē
end

