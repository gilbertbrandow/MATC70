import numpy as np
import scipy
import scipy.linalg

from testmatrices import testmatrix


def heltalsev(A, tol=0.0001):
    """Decide if the matrix A has integer eigenvalues.
    
    Returns (ev, mult)
    Raises ValueError if some eigenvalue is not within tol of an integer.
    """
    values = np.linalg.eigvals(A)
    rounded = np.round(values.real).astype(int)
    distances = np.abs(values - rounded)
    
    # If one of the distances exceeds our tolerance we raise an error
    if np.any(distances >= tol):
        raise ValueError(f"not all eigenvalues are integers")

    return np.unique(rounded, return_counts=True)


def jordan_matrix(A, tol=0.00001):
    """Compute the Jordan normal form of a matrix A with integer eigenvalues.

    Uses Theorem 7.9: for an eigenvalue λ, let p_k = dim Ker(A - λI)^k and
    n_k = number of Jordan blocks J_k(λ) of size k. Then

        p_1 = n_1 +  n_2 +  n_3 + ...
        p_2 = n_1 + 2n_2 + 2n_3 + ...
        p_3 = n_1 + 2n_2 + 3n_3 + ...

    Subtracting consecutive equations gives p_k - p_(k-1) = n_k + n_(k+1) + ...
    (the number of blocks of size >= k), and subtracting once more gives

        n_k = 2p_k - p_(k-1) - p_(k+1),   with p_0 = 0.
    """
    ev, mult = heltalsev(A, tol)
    A = np.array(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n)

    blocks = []
    for eigenvalue, m in zip(ev, mult):
        if m == 1:
            blocks.append(np.array([[eigenvalue]]))
            continue

        B = A - eigenvalue * I
        p = [0]
        power = I
        while p[-1] < m and len(p) <= m:
            power = power @ B
            p.append(scipy.linalg.null_space(power).shape[1])
        p.append(p[-1])  # p_(k+1) = p_k once the kernel is complete

        for k in range(1, len(p) - 1):
            n_k = 2 * p[k] - p[k - 1] - p[k + 1]
            block = eigenvalue * np.eye(k, dtype=int) + np.eye(k, k=1, dtype=int)
            blocks.extend([block] * n_k)

    return scipy.linalg.block_diag(*blocks)


def main():
    for i in range(0, 7):
        A = testmatrix(i)
        try:
            print(f"testmatrix({i}):")
            print(jordan_matrix(A))
        except ValueError as e:
            print(f"  {e}")



if __name__ == "__main__":
    main()
