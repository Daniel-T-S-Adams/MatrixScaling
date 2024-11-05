# Imports from the standard library
import numpy as np
from typing import Tuple


def generate_parameters(min_n : int, max_n : int, min_m : int, max_m : int, min_w : float, max_w : float, min_epsilon : float, max_epsilon : float) -> Tuple[int, int, float, float]:
    """
    Generates a the parameters for our problem. Uniformly sampled from the intervals min_ , max_
    
    Parameters:
       min_n, max_n (int): The range of the number of rows in matrix A.
       min_m, max_n (int): The range of the number of columns in matrix A.
       min_w, max_w (float): The range of the sum to which the vectors r and c should scale.
       min_epsilon, max_epsilon (float): The range of the error bound epsilon.
    
    Returns:
        Tuple[int, int, float, float]: A tuple containing the number of rows, number of columns, sum to which the vectors r and c should scale, and the error bound epsilon.
    """
    
    # Generate the number of rows n
    n = np.random.randint(min_n, max_n)
    # Generate the number of columns m
    m = np.random.randint(min_m, max_m)
    # Generate the sum to which the vectors r and c should scale
    w = np.random.uniform(min_w, max_w)
    # Generate the error bound epsilon
    epsilon = np.random.uniform(min_epsilon, max_epsilon)
    
    return n, m, w, epsilon



def generate_matrix_and_vectors(
    n: int, m: int, w: float, A_max : float, r_max : float, c_max : float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generates a random matrix A and vectors r and c for the matrix scaling problem.
    The Values of the Matrix are between 0 and A_max
    The Values of r are between 0 and r_max
    The Values of c are between 0 and c_max
    
    Parameters:
        n (int): Number of rows in matrix A.
        m (int): Number of columns in matrix A.
        A_max (float): Maximum value for the entries of matrix A.
        r_max (float): Maximum value for the entries of vector r.
        c_max (float): Maximum value for the entries of vector c.
        w (float): Sum to which the vectors r and c should scale.
        
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing matrix A, vectors r, c.
    """
    # Avoid zeros by setting a minimum value
    min_value = 1e-10

    # Generate matrix A with entries sampled from Uniform(min_value, 1)
    A = np.random.uniform(min_value, A_max, (n, m))

    # Generate vector r and normalize to sum to w
    r_raw = np.random.uniform(min_value, r_max, n)
    r = (r_raw / r_raw.sum() * w).reshape((n, 1))

    # Generate vector c and normalize to sum to w
    c_raw = np.random.uniform(min_value, c_max, m)
    c = (c_raw / c_raw.sum() * w).reshape((m, 1))
    
    return A, r, c