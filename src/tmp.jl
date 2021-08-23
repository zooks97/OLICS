using LinearAlgebra

function main()
    T = Float64

    nα = 3
    M̄ = Array{T}(undef, 6, 6, nα)
    M̄[:,:,1] = [
        1 0 0 0 0 0
        0 1 0 0 0 0
        0 0 1 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
    ]
    M̄[:,:,2] = [
        0 1 1 0 0 0
        1 0 1 0 0 0
        1 1 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
        0 0 0 0 0 0
    ]
    M̄[:,:,3] = [
        0 0 0 0 0 0
        0 0 0 0 0 0 
        0 0 0 0 0 0
        0 0 0 1 0 0
        0 0 0 0 1 0
        0 0 0 0 0 1
    ]
    c = Vector{T}(undef, nα)
    c = [104, 73, 32]

    C = zeros(T, (6, 6))
    @inbounds begin
        for α in 1:nα
            C += M̄[:,:,α] * c[α]
        end
    end

    nγ = 2
    η = Matrix{T}(undef, 6, nγ)
    η[:,1] = normalize([ 1  2  3  4  5  6])
    η[:,2] = normalize([ 3 -5 -1  6  2 -4])

    Ē = Matrix{T}(undef, 6 * nγ, nα)
    @inbounds begin
        for α in 1:nα
            for γ in 1:nγ
                Ē[(6 * γ - 5):(6 * γ), α] = M̄[:,:,α] * η[:,γ]
            end
        end
    end

    a = Vector{T}(undef, 6 * nγ)
    @inbounds begin
        for γ in 1:nγ
            a[(6 * γ - 5):(6 * γ)] = C * η[:,γ]
        end
    end

    ĉ = Ē \ a

    D_opt = 1 / det(transpose(Ē) * Ē)

end