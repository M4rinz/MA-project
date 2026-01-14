using LinearAlgebra, Symbolics

@variables a b c d e f

# Slightly expanded example w.r.t. the one in the article
L = zeros(Num, 7, 7);
L[1:2,3] = [a; b];
L[3,4] = c;
L[4,5] = d;
L[5,6:7] = [e f];
L = L';
print("Adjacency matrix L' = \n")
display(L')
print("\n")

invImLp = inv(I(7) - L');
print("(I - L')⁻¹ = \n")
display(invImLp)
print("\n")

print("Path weights from sources to sinks are given by\n")
print("\t[I(2); zeros(5,2)]' * (I - L')⁻¹ * [zeros(6,1); 1] = \n")
display(invImLp[1:2, end-1:end])
print("\n\n")

print("What if we relabeled nodes 1 and 3 by swapping them, leaving the graph unchanged?\n")
T = zeros(Num, 7,7);
T[1, 4] = c;
T[2:3, 1] = [b; a];
T[4:end, :] = L'[4:end, :];
T = T';
print("Adjacency matrix of the new graph T' = \n")
display(T')
print("\n")

invImTp = inv(I(7) - T');
print("(I - T')⁻¹ = \n")
display(invImTp)
print("\n")

print("Path weights from sources to sinks are now given by\n")
print("\t(I - L')⁻¹[ [3,2], 6:7] = \n")
display(invImTp[[3,2], end-1:end])
