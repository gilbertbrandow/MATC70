import numpy as np
from sympy import Matrix

def testmatrix(n):
    # ett antal matriser att prova jordan program på

    if n < 0 or n > 6: raise AssertionError('Argumentet n ska vara heltal mellan 0 och 6')
    match n:
        case 0:
            M = np.zeros((3,3))
        case 1:
            M = np.eye(4)
        case 2:
            M = [[1, 1, 1], [1, 1, 1], [-2, -2, -2]]; 
        case 3:
            # Troligen klarar programmet inte det trots att alla egenvärden är 0, men
            # Python kommer formodligen inte klara att upptäcka det. Ger felmeddelandet i stället.
            M = [[-9,   11, -21,    63, -252],
                [70, -69, 141,  -421, 1684],
                [-575, 575, -1149, 3451, -13801],
                [3891, -3891, 7782, -23345, 93365],
                [1024, -1024, 2048, -6144, 24572]]; 
        case 4:
            # Given testmatris
            coeffs = np.poly(list(range(1, 11)))
            M = np.polynomial.polynomial.polycompanion(coeffs[::-1])
        case 5:
            # Klarar ni komplexa tal?
            M = [[3, -4], [4, 3]]
        case 6:
            # Testa hur klarar programet fel matris med olika tolerans.
            M = np.array([[1000, 1], [1, 0]])
    return M
