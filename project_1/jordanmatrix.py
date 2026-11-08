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
    """
    ev, mult = heltalsev(A, tol)
    raise NotImplementedError


def main():
    for i in range(0, 7):
        A = testmatrix(i)
        try:
            ev, mult = heltalsev(A)
            print(f"testmatrix({i}): ev = {ev}, mult = {mult}")
        except ValueError as e:
            print(f"testmatrix({i}): {e}")



if __name__ == "__main__":
    main()
