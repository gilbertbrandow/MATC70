import numpy as np
import scipy
import scipy.linalg

from testmatrices import testmatrix


def heltalsev(A, tol=0.0001):
    """Decide if the matrix A has integer eigenvalues.
    
    Returns (eigenvalues, multiplicities)
    Raises ValueError if some eigenvalue is not within tol of an integer.
    """
    values = np.linalg.eigvals(A)
    rounded = np.round(values.real).astype(int)
    distances = np.abs(values - rounded)
    
    # If one of the distances exceeds our tolerance we raise an error
    if np.any(distances >= tol):
        raise ValueError("not all eigenvalues are integers")

    return np.unique(rounded, return_counts=True)


def jordanmatris(A, tol=0.00001):
    """Compute the Jordan normal form of a matrix A with integer eigenvalues.
    
    Returns the jordan matrix
    Raises ValueError if some eigenvalue is not within tol of an integer.
    """
    ev, mult = heltalsev(A, tol)
    n = len(A)
    I = np.eye(n)
    
    # Create the empty jordan matrix to which we will append all the blocks
    J = np.zeros((0, 0), dtype=int)

    # Looping over all eigenvalues and multiplicites
    for lam, m in zip(ev, mult):
        B = A - lam * I

        # Starting values for each eigenvalue
        p = [0]
        power = I
        
        # Computing all p_k values up until multiplicites
        while p[-1] < m and len(p) <= m:
            power = power @ B
            # Using rank-nullity theorem (5.12) we compute p_k=dim ker B^k=n-rank(B^k) 
            p.append(n - np.linalg.matrix_rank(power))
        
        # p_k must reach m; if not, λ is not an exact eigenvalue (tol too large)
        if p[-1] != m:
            raise ValueError(f"dim Ker(A - {lam}I)^k never reached {m}")

        # p_k can never exceed m, so once p_k = m we also have p_(k+1) = m.
        # We add that value so at_least ends with 0, without it the largest block size would be lost.
        p.append(m)

        # Theorem 7.9: p_k - p_(k-1) = n_k + n_(k+1) + ...
        # at_least[k-1] = number of Jordan blocks of size >= k
        at_least = np.diff(p)

        # (size >= k) - (size >= k+1) = n_k, the number of blocks of size exactly k.
        exactly = -np.diff(at_least)

        # for our fixed eigenvalue lambda, k is the size of the jordanblock and n_k the number of such blocks
        for k, n_k in enumerate(exactly, start=1):
            # Create the jordanblock of size k for our eigenvalue lambda
            jordanblock_k = lam * np.eye(k, dtype=int) + np.eye(k, k=1, dtype=int)
            for _ in range(n_k):
                # append the jordanblock_k n_k times to our jordan matrix J
                J = scipy.linalg.block_diag(J, jordanblock_k)

    return J


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
