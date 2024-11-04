# Main file for running the code#

# imports from libraries 

# imports from local files
from GenerateProblem import Generate_matrix_and_vectors_and_error
from Baseline import Baseline_algorithm
from VerifySolution import Verify

# generate the parameters
n , m, w, min_epsilon, max_epsilon = 1000, 1000, 1, 0.0001, 0.001
print("Parameters Generated")

# generate the problem 
A, r, c, epsilon = Generate_matrix_and_vectors_and_error(n , m, w, min_epsilon, max_epsilon)
print("Problem Generated") 
print(f"A has type {type(A)} and shape {A.shape}")
print(f"r has type {type(r)} and shape {r.shape}")
print(f"c has type {type(c)} and shape {c.shape}")
print(f"epsilon is equal to {epsilon}")

# solve the problem
u, v = Baseline_algorithm(A, r, c, epsilon, max_iter = 10**10)
print("Problem Solved")
print(f"u has type {type(u)} and shape {u.shape}")
print(f"v has type {type(v)} and shape {v.shape}")

# verify the solution
print("Solution Verified")
print(Verify(A, r, c, u, v, epsilon))


