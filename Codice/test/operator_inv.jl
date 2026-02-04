module MyOperator

using LinearAlgebra

## Custom datatype representing linear invertible operators
struct Operator
    op
    adj 
    inv
    adj_inv
    sym
end


# "Overloading" of operations
import Base:  zero, one, show, adjoint, *, \, ∘, +, -
show(io::IO, M::Operator) = print(io, M.sym)  # pretty printing
zero(::Any) = 𝒪()   # Let's make any undefined zero the 𝒪 operator
one(::Any) = ℐ()    # Let's make any undefined one the ℐ operator

# Adjoints
adjoint(A::Operator) = Operator(A.adj, A.op, A.adj_inv, A.adj, "("*A.sym*")'")
adjoint(B::Bidiagonal) = Bidiagonal(adjoint.(B.dv),adjoint.(B.ev),(B.uplo == 'U') ? :L : :U) # lower to upper

# Operations
-(A::Operator) = Operator(X->-A.op(X), X->-A.adj(X), X->-A.inv(X), X->-A.adj_inv(X), "-"*A.sym)
-(::typeof(𝒪), X::Matrix) = -X          # 𝒪 - X should be -X
+(A::Operator, B::Operator) = Operator(
                                X -> (A.op(X) + B.op(X)), 
                                X -> (A.adj(X) + B.adj(X)), 
                                X -> (A.inv),
                                A.sym*" + "*B.sym)


*(A::Operator, X::Matrix) = A.op(X)     # Operator * Matrix means evaluating the operator
\(ℐ::typeof(ℐ()), A::Matrix) = A
∘(A::Operator, B::Operator) = Operator(A.op ∘ B.op, B.adj ∘ A.adj, A.sym*"∘"*B.sym)
*(A::Operator, B::Operator) = A ∘ B     # Product between operators is their composition
*(M::Adjoint{Operator, Matrix{Operator}},v::Array) = M .* [v]   # We need [A;B]*C to somehow magically be [AC;BC]
+(A::Array,x::Number)=A.+x



# Operators
Left_mul(A::Matrix) = Operator(
                        X -> A * X,
                        X -> A \ X,
                        X -> A' * X,
                        "ℒ$(size(A))"
                    )

ℐ() = Operator(X->X, X->X, X->X, "I")     # identity operator
𝒪() = Operator(X->zero(X), :ND, X->zero(X), "𝒪")# zero operator




end

