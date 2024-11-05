# Imports from the standard library
import numpy as np
import logging

def verify(
    A: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    epsilon: float
) -> bool:
    """
    Verifies the solution of the matrix scaling problem.
    
    Parameters:
        A (np.ndarray): The original matrix.
        r (np.ndarray): Target row sums vector.
        c (np.ndarray): Target column sums vector.
        u (np.ndarray): Scaling vector for rows.
        v (np.ndarray): Scaling vector for columns.
        epsilon (float): Error tolerance.
    
    Returns:
        bool: True if the solution meets the error tolerance, False otherwise.
    """
    # Validate inputs
    n, m = A.shape
    assert r.shape == (n, 1), f"Expected r shape ({n}, 1), got {r.shape}"
    assert c.shape == (m, 1), f"Expected c shape ({m}, 1), got {c.shape}"
    assert u.shape == (n, 1), f"Expected u shape ({n}, 1), got {u.shape}"
    assert v.shape == (m, 1), f"Expected v shape ({m}, 1), got {v.shape}"
    assert isinstance(epsilon, float), f"epsilon must be a float, got {type(epsilon)}"

    # Calculate scaled matrix
    scaled_A = (u * A) * v.T

    # Calculate row and column sums
    row_sums = scaled_A.sum(axis=1, keepdims=True)
    col_sums = scaled_A.sum(axis=0, keepdims=True).T

    # Compute relative errors
    error_r = np.linalg.norm(row_sums - r, ord=1) 
    error_c = np.linalg.norm(col_sums - c, ord=1) 

    # Check if errors are within tolerance
    is_verified = error_r < epsilon and error_c < epsilon

    # Logging the verification result
    logging.debug(f"Row error: {error_r}, Column error: {error_c}, Verified: {is_verified}")

    return is_verified
