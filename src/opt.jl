using LinearAlgebra
using JLD
using Optim
using Random

function get_ulics(T::Type, laue::AbstractString)::Matrix{T}
    ulics = JLD.load("ULICS.jld")
    return convert.(T, ulics[laue])
end

function get_elastic_symmetries(T::Type, laue::AbstractString)::Array{T, 3}
    symmetries = JLD.load("elastic_symmetries.jld")
    return convert.(T, symmetries[laue])
end

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

function to_cartesian(y)
    r = y[1]
    ϕ = y[2:end]
    return to_cartesian(r, ϕ)
end

function to_cartesian(r, ϕ)
    n = length(ϕ) + 1
    x = typeof(ϕ)(undef, n)
    @inbounds begin
        x[1] = r * cos(ϕ[1])
        for i in 2:n-1
            x[i] = r * prod(sin.(ϕ[1:i-1])) * cos(ϕ[i])
        end
        x[n] = r * prod(sin.(ϕ[1:n-1]))
    end
    return x
end

function construct_expriment_matrix(
    η::AbstractMatrix{T},
    M̄::AbstractArray{T, 3}
)::AbstractMatrix{T} where T <: Real
    ndim = size(η, 1)
    nγ = size(η, 2)
    nα = size(M̄, 3)
    Ē = Matrix{T}(undef, (ndim * nγ, nα))
    @inbounds begin
        for α in 1:nα
            for γ in 1:nγ
                Ē[(ndim * γ - ndim + 1):(ndim * γ), α] = M̄[:,:,α] * η[:,γ]
            end
        end
    end
    return Ē
end

function result_to_cartesian(result, nγ)
    ndim = div(size(result, 1), nγ) + 1
    result = reshape(result, (ndim - 1, nγ))
    x = Matrix{Float64}(undef, (ndim, nγ))
    for γ in 1:nγ
        x[:,γ] = to_cartesian(1.0, reshape(result[:,γ], ndim - 1))
    end
    return x
end

function target(η_spherical)
    ndim = 6
    nγ = 1
    M̄ = get_elastic_symmetries(Float64, "CI")

    η_spherical = reshape(η_spherical, (ndim - 1, nγ))
    η_cartesian = Matrix{Float64}(undef, (ndim, nγ))
    @inbounds for γ in 1:nγ
        η_cartesian[:,γ] = to_cartesian(1.0, η_spherical[:,γ])
    end
    
    Ē = construct_expriment_matrix(η_cartesian, M̄)
    return 1 / max(1e-12, abs(det(transpose(Ē) * Ē)))
end

function main()
    ndim = 6
    nγ = 1

    lower = [0., 0., 0., 0., 0.]
    upper = [1π, 1π, 1π, 1π, 2π]

    η₀_spherical = Matrix{Float64}(undef, (ndim - 1, nγ))  
    for γ in 1:nγ
        η₀_spherical[:,γ] = (rand(Float64, ndim - 1) .+ lower) .* upper
    end
    η₀_spherical = reshape(η₀_spherical, (ndim - 1 * nγ))

    result = optimize(target, repeat(lower, nγ), repeat(upper, nγ),
                      η₀_spherical, Fminbox(LBFGS()),
                      Optim.Options(g_tol=1e-14, iterations=100))
    return result
end

result = main()