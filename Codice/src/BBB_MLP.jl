module BBB_MLP

include("Operator.jl")

using LinearAlgebra, .Operator, OffsetArrays

# activation functions
h(x)  = tanh(x)
h′(x) = 1 - h(x)^2

function forward_pass(params,X₀;h=h,h′= h′)
    T = Matrix{Float64}
    N = length(params)
    X = OffsetArray(Vector{T}(undef,N+1),0:N)   
    Δ = Vector{T}(undef, N)
    X[0] = X₀
    W = first.(params)
    B = last.(params)
    
    for i=1:N         
          X[i] =  h.(W[i]*X[i-1] .+ B[i])
          Δ[i] =  h′.(W[i]*X[i-1] .+ B[i])        
    end 
    X,Δ
end

# Loss function and its gradient (w.r.t. prediction)
𝓁(x,y)  = sum(abs2, x-y) / 2
𝓁′(x,y) = x .- y;


init(sizes...) = 0.01randn(sizes...)

function create_Ws_and_bs(n=[5,4,3,1])
    N = length(n) - 1
    Ws_and_bs =[ [init(n[i+1],n[i]) , init(n[i+1])]  for i=1:N]

    return Ws_and_bs
end


function create_X_δ(Ws_and_bs; n=[5,4,3,1], k=10)
    # parameters
    #N = length(n) - 1

    ## weights and biases of the MLP
    #Ws_and_bs =[ [init(n[i+1],n[i]) , init(n[i+1])]  for i=1:N]

    # create dataset
    X₀ = init(n[1],k)       # patterns

    X, δ = forward_pass(Ws_and_bs,X₀)
    return X, δ
end

function create_op_matrices(X, δ, Ws_and_bs)
    N = length(δ)       
    k = size(X[0], 2) # read batchsize from input data

    # create labels at random
    y = init(size(X[end], 2), k)

    M = Diagonal([ [ℋ(δ[i]) ∘ ℛ(X[i-1])  ℋ(δ[i]) ∘ ℛ(ones(1,k))] for i=1:N])
    ImL = Bidiagonal([ℐ() for i in 1:N], -[ℋ(δ[i]) ∘ ℒ(Ws_and_bs[i][1]) for i=2:N] , :L)

    g = [ fill(𝒪(),N-1) ; [𝓁′(X[N],y)] ]  

    return M, ImL, g
end



export create_Ws_and_bs, create_X_δ, create_op_matrices

end #module

#using .BBB_MLP