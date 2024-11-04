import numpy as np
from typing import Tuple

def Baseline_algorithm(A: np.ndarray, r: np.ndarray, c: np.ndarray, epsilon: float, max_iter: int = int(1e10)) -> Tuple[np.ndarray, np.ndarray]:
    n, m = A.shape

    # Initialize scaling vectors
    u = np.ones((n, 1))
    v = np.ones((m, 1))
    tiny_delta = 1e-5

    for i in range(max_iter):
        # Update v and u iteratively
        v = c / (A.T @ u + tiny_delta)
        u = r / (A @ v + tiny_delta)

        # Calculate errors for debugging
        row_sums = u * (A @ v)
        col_sums = v * (A.T @ u)
        error_r = np.linalg.norm(row_sums - r, ord=1)
        error_c = np.linalg.norm(col_sums - c, ord=1)
        
        # Debugging output for errors
        if i % 100 == 0:
            print(f"Baseline Iteration {i}: error_r = {error_r}, error_c = {error_c}")

        if error_r < epsilon and error_c < epsilon:
            print("Convergence achieved.")
            break
    else:
        print("Warning: Maximum iterations reached without convergence")


    return u, v
