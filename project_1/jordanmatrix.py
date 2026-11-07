import numpy as np
import scipy
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

from testmatrices import testmatrix


def integer_eigenvalues(
    A: ArrayLike, tol: float = 0.0001
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Decide if the matrix A has integer eigenvalues.

    Returns (ev, mult) where ev is a vector of the distinct eigenvalues
    and mult is a vector with the corresponding multiplicities.
    """
    raise NotImplementedError


def jordan_matrix(A: ArrayLike, tol: float = 0.00001) -> NDArray[np.int64]:
    """Compute the Jordan normal form of a matrix A with integer eigenvalues.
    """
    ev, mult = integer_eigenvalues(A, tol)
    raise NotImplementedError


def main() -> None:
    for i in range(0, 7):
        A = testmatrix(i)
        print(jordan_matrix(A))



if __name__ == "__main__":
    main()
