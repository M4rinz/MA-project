using LinearAlgebra, Symbolics

@variables a b s

L = zeros(Num, 3,3);
L[1:2,2] = [a;s];
L[2,end] = b;
L = L';

print("Adjacency matrix L' = \n")
display(L')
print("\n")

invImLp = inv(I(3) - L');
print("(I - L')⁻¹ = \n")
display(invImLp)
print("\n")

print("Notice that (1 - s) * (I - L')⁻¹ = \n")
display((1 - s) * invImLp)
print("\n")