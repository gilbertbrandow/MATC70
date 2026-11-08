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


def jordanmatris(A, tol=0.00001):
    """Compute the Jordan normal form of a matrix A with integer eigenvalues.
    """
    ev, mult = heltalsev(A, tol)
    n = A.shape[0]
    I = np.eye(n)

    blocks = []
    for lam, m in zip(ev, mult):
        B = A - lam * I

        p = [0]
        power = I
        while p[-1] < m and len(p) <= m:
            power = power @ B
            p.append(n - np.linalg.matrix_rank(power, tol=tol))
        if p[-1] != m:
            raise ValueError(f"kernel dimensions for eigenvalue {lam} never "
                             f"reached the multiplicity {m}: p = {p}")
        p.append(m)
        at_least = np.diff(p)
        exactly = -np.diff(at_least)

        for k, n_k in enumerate(exactly, start=1):
            J_k = lam * np.eye(k, dtype=int) + np.eye(k, k=1, dtype=int)
            blocks.extend([J_k] * n_k)

    return scipy.linalg.block_diag(blocks)


def main():
    for i in range(0, 7):
        A = testmatrix(i)
        try:
            print(f"testmatrix({i}):")
            print(jordanmatris(A))
        except ValueError as e:
            print(f"  {e}")



if __name__ == "__main__":
    main()
