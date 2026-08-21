import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    m = len(A)
    n = len(A[0])

    T = np.zeros((n,m), dtype=type(A[0][0]))

    for i in range(m):
        for j in range(n):
            T[j][i] = A[i][j]

    return T
