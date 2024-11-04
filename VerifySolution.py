import numpy as np

def Verify(A: np.ndarray, r: np.ndarray, c: np.ndarray, u: np.ndarray, v: np.ndarray, epsilon: float) -> bool:

    ### Input type and size check ###
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise TypeError(f"A must be a 2D NumPy array, but got {type(A)} with shape {A.shape if isinstance(A, np.ndarray) else 'N/A'}")

    # Get dimensions of A for checking other inputs
    n, m = A.shape

    # Check r and u for shape (n, 1)
    for name, vector in {"r": r, "u": u}.items():
        if not isinstance(vector, np.ndarray) or vector.shape != (n, 1):
            raise TypeError(f"{name} must be a NumPy array with shape ({n}, 1), but got {type(vector)} with shape {vector.shape if isinstance(vector, np.ndarray) else 'N/A'}")

    # Check c and v for shape (m, 1)
    for name, vector in {"c": c, "v": v}.items():
        if not isinstance(vector, np.ndarray) or vector.shape != (m, 1):
            raise TypeError(f"{name} must be a NumPy array with shape ({m}, 1), but got {type(vector)} with shape {vector.shape if isinstance(vector, np.ndarray) else 'N/A'}")

    # Check epsilon is a float
    if not isinstance(epsilon, float):
        raise TypeError(f"epsilon must be a float, but got {type(epsilon)}")


    # Calculate row and column sums using the final values of u and v
    row_sums = u * (A @ v)
    col_sums = v * (A.T @ u)
    error_r = np.linalg.norm(row_sums - r, ord=1)
    error_c = np.linalg.norm(col_sums - c, ord=1)
    

    print(f"Check 1 (error_r < epsilon): {error_r < epsilon}")
    print(f"Check 2 (error_c < epsilon): {error_c < epsilon}")

    return error_r < epsilon and error_c < epsilon

