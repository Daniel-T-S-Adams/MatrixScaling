# Imports from the standard library
import logging
import sys

# Imports from other files
from GenerateProblem import generate_matrix_and_vectors
from GenerateProblem import generate_parameters
from Baseline import baseline_algorithm
from VerifySolution import verify

def main():
    """
    Main function to generate a matrix scaling problem, solve it, and verify the solution.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Generate parameters for the problems. 
    min_n, max_n = 5000, 5001 # the number of rows and columns in the matrix A is sampled uniformly between min_n and max_n
    min_m, max_m = 5000, 5001 # the number of rows and columns in the matrix A is sampled uniformly between min_n and max_n
    min_w, max_w = 1, 10  # the sum to which the vectors r and c should scale is sampled uniformly between min_w and max_w
    min_epsilon, max_epsilon = 0.0001, 0.00001 # error tolerance, the error tolerance is sampled uniformly between min_epsilon and max_epsilon
    A_max = 4  # components of A are sampled uniformly between 0 and A_max
    r_max = 3  # components of r are sampled uniformly between 0 and r_max
    c_max = 2  # components of c are sampled uniformly between 0 and c_max
    
    n, m, w, epsilon = generate_parameters(min_n, max_n, min_m, max_m, min_w, max_w, min_epsilon, max_epsilon)
    
    logging.info("Parameters generated")

    # Generate the problem
    A, r, c = generate_matrix_and_vectors(n, m, w, A_max, r_max, c_max)
    logging.info("Problem generated")
    logging.debug(f"A shape: {A.shape}, r shape: {r.shape}, c shape: {c.shape}, epsilon: {epsilon}")
    
    # Solve the problem
    u, v = baseline_algorithm(A, r, c, epsilon, max_iter=1_000_000)
    logging.info("Problem solved")
    logging.debug(f"u shape: {u.shape}, v shape: {v.shape}")

    # Verify the solution
    is_verified = verify(A, r, c, u, v, epsilon)
    logging.info(f"Solution verified: {is_verified}")

if __name__ == "__main__":
    main()
