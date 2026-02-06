module MyOperator

using LinearAlgebra

struct Operator  # Linear Matrix Operators from Matrices to Matrices (and the operator adjoint)
    op
    adj
    sym
end

# left multiply by A (X → AX)
ℒ(A::Matrix) = Operator(X -> A*X, X -> A'*X, "ℒ$(size(A))")  

# right multiply by A (X → XA)
ℛ(A::Matrix) = Operator(X->X*A, X->X*A', "ℛ$(size(A))")  

# Hadamard (aka elementwise) product
ℋ(A::Matrix) = Operator(X->X.*A, X->X.*A, "ℋ$(size(A))")  

# identity operator
ℐ() = Operator(X->X, X->X, "I")  

# zero operator
𝒪() = Operator(X->zero(X), X->zero(X), "𝒪") 

export ℒ, ℛ, ℋ, ℐ, 𝒪


import Base: zero, one, show

show(io::IO, M::Operator) = print(io, M.sym)  # pretty printing
zero(::Any) = 𝒪()   
#zero(::Operator) = 𝒪() 
one(::Operator) = ℐ()      

## Adjoints
import Base: adjoint

adjoint(A::Operator) = Operator(A.adj, A.op, "("*A.sym*")'")
adjoint(B::Bidiagonal) = Bidiagonal(adjoint.(B.dv),
                                    adjoint.(B.ev),
                                    (B.uplo == 'U') ? :L : :U) # lower to upper

## arithmetic operations
import Base: *, \, ∘, +, -

-(A::Operator) = Operator(X->-A.op(X), X->-A.adj(X), "-"*A.sym)
-(::typeof(𝒪()), X::Matrix) = -X          # 𝒪 - X should be -X
+(A::Operator, B::Operator) = Operator(
                                X -> (A.op(X) + B.op(X)), 
                                X -> (A.adj(X) + B.adj(X)), 
                                A.sym*" + "*B.sym)
+(::typeof(𝒪()), X::Operator) = X   # summing the zero operator is ignored
-(A::Operator, B::Operator) = A + (-B)

\(ℐ::typeof(ℐ()), A::Matrix) = A
∘(A::Operator, B::Operator) = Operator(A.op ∘ B.op, 
                                        B.adj ∘ A.adj, 
                                        A.sym*"∘"*B.sym)

# The product between operators is their composition                                        
*(A::Operator, B::Operator) = A ∘ B     

# We need [A;B]*C to somehow magically be [AC;BC]
*(M::Adjoint{Operator, Matrix{Operator}}, v::Array) = M .* [v] 

# Operator * Matrix means evaluating the operator
*(A::Operator, X::Matrix) = A.op(X)     
+(A::Array,x::Number)=A.+x

# Ci serve di poter fare il broadcast di un operatore a un array di matrici
# Per fortuna Julia è molto furbo e ci permette di farlo EASY
*(A::Operator, V::Array{Matrix}) = A.op.(V)

# la martellata suprema
#*(::typeof(𝒪()), X::Matrix) = 𝒪()
#+(::typeof(𝒪()), X::Matrix) = X



end #module

#using .Operator