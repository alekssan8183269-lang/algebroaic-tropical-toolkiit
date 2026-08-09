import numpy as np
from numba import njit

@njit(fastmath=True)
def numba_tropical_matrix_mul(A, B, zero_val):
    """
    Сверхбыстрое тропическое умножение матриц (Max-Plus) на голом Си-скоростях.
    Использует JIT-компиляцию Numba для обхода ограничений Python.
    """
    n = A.shape[0]
    m = B.shape[1]
    p = A.shape[1]
    
    C = np.full((n, m), zero_val, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            max_val = zero_val
            for k in range(p):
                if A[i, k] != zero_val and B[k, j] != zero_val:
                    current_val = A[i, k] + B[k, j]
                    if current_val > max_val:
                        max_val = current_val
            C[i, j] = max_val
            
    return C
