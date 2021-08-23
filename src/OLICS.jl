using LinearAlgebra
using JLD

module OLICS

    function construct_ulics(T::AbstractType)::Matrix{T}
        ulics = [
            1 -2  3 -4  5  6
            2  1 -5 -6  4  3
            3  4 -1  5  6 -2
            4 -3  6  1 -2  5
            5  6  2 -3 -1 -4
            6 -5 -4  2 -3  1
        ]
        ulics = ulics ./ sqrt.(sum(ulics.^2, dims=1))
        ulics = Dict(
            "CI"  => ulics[:,[1]],
            "CII" => ulics[:,[1]],
            "HI"  => ulics[:,[1,3]],
            "HII" => ulics[:,[1,3]],
            "RI"  => ulics[:,[1,3]],
            "RII" => ulics[:,[1,3]],
            "TI"  => ulics[:,[1,3]],
            "TII" => ulics[:,[1,3]],
            "O"   => ulics[:,[1,3,5]],
            "Mb"  => ulics[:,[1,2,3,4,5]],
            "Mc"  => ulics[:,[1,2,3,4,5]],
            "N"   => ulics
        )
        return convert.(T, ulics)
    end

    function get_ulics(T::AbstractType, laue::AbstractString)::Matrix{T}
        ulics = JLD.load("ULICS.jld")
        return convert.(T, ulics[laue])
    end

    function get_elastic_symmetries(T::AbstractType, laue::AbstractString)::Array{T, 3}
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
    
    function construct_expriment_matrix(
        η::AbstractMatrix{T},
        M̄::AbstractArray{T, 3}
    )::AbstractMatrix{T} where T <: Real
        nγ = size(η, 2)
        Ē = Matrix{T}(undef, (6 * nγ, nα))
        @inbounds begin
            for α in 1:nα
                for γ in 1:nγ
                    Ē[(6 * γ - 5):(6 * γ), α] = M̄[:,:,α] * η[:,γ]
                end
            end
        end
        return Ē
    end
    
    function D_cost(η₀::AbstractMatrix, M̄::AbstractArray, nγ::Integer)
        η = reshape(η₀, (6, nγ))
        Ē = construct_expriment_matrix(η, M̄)
        return 1 / max(1e-12, abs(det(Ē' * Ē)))
    end

    function D_cost(η₀::AbstractMatrix, M̄::AbstractArray, nγ::Integer, c::AbstractVector)
        η = reshape(η₀, (6, nγ))
        Ē = construct_expriment_matrix(η, M̄)
        return 1 / max(1e-12, abs(det(Ē' * Ē * (c * c'))))
    end

end # module OLICS