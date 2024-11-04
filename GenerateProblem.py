import numpy as np
from typing import Tuple


# Input dimensions of matrix A, vectors r and c, and the sums of the vectors w.
# Also min_epsilon & max_epsilon give the upper bounds on the error allowed.
def Generate_matrix_and_vectors_and_error(
    n: int, m: int, w: float, min_epsilon: float, max_epsilon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    
    # Generate n x m matrix A with entries sampled from Uniform(0,1)
    # Add a small offset to avoid zero
    A = np.random.uniform(0, 1, (n, m))
    A[A == 0] = np.nextafter(0, 1)

    # Generate vector r of length n and sum w, then reshape to (n, 1)
    r_raw = np.random.uniform(0, 1, n)
    r_raw[r_raw == 0] = np.nextafter(0, 1)
    r = ((r_raw / r_raw.sum()) * w).reshape((n, 1))

    # Generate vector c of length m and sum w, then reshape to (m, 1)
    c_raw = np.random.uniform(0, 1, m)
    c_raw[c_raw == 0] = np.nextafter(0, 1)
    c = ((c_raw / c_raw.sum()) * w).reshape((m, 1))

    # Generate the error bound
    epsilon = np.random.uniform(min_epsilon, max_epsilon)

    return A, r, c, epsilon
