restart;
?LinearAlgebra
with(LinearAlgebra):

x := 7;
y := 6;

?Matrix
A := Matrix([[1, x, 0], [1, y, x], [0, -1, 1]]);
Id := IdentityMatrix(3);

# 1
Determinant(A);
Trace(A);
Transpose(A);
MatrixInverse(A);
(A + Id) . (Transpose(A) - Id);

# 2
Eigenvalues(A);
Eigenvectors(A);

# 3
?JordanForm
J, Q := JordanForm(A, output = ['J', 'Q']);
simplify(Q . J . MatrixInverse(Q) - A);

# 4
Rank(A);
Rank(A - Id);

# 5
p := CharacteristicPolynomial(A, t);
factor(p);

# 6
MinimalPolynomial(A, t);

# 7
NullSpace((A - Id)^2);
ColumnSpace(A - Id);

# 8
?GramSchmidt
GramSchmidt(ColumnSpace(A - Id), normalized = true);

# 9
MatrixExponential(A);

# 10
P, L, U := LUDecomposition(A);
Qr, R := QRDecomposition(A);

# 11
Norm(A, 2);
evalf(%);
ConditionNumber(A, 2);
evalf(%);

# 12
?MatrixFunction
C := MatrixFunction(A, sqrt(t), t);
simplify(C . C - A);