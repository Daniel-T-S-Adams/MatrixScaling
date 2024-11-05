# test script
import logging
from Baseline import baseline_algorithm
from VerifySolution import verify
import numpy as np

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

A = np.array([[1,2,1],[2,1,4],[1,5,2]])
# make sure r and c sum is equal. 
r = np.array([1 , 5 , 3]).reshape(-1,1) 
c = np.array([1 , 4 , 4]).reshape(-1,1)

u,v = baseline_algorithm(A, r , c , 0.01)

row_sums = u * (A @ v)
col_sums = v * (A.T @ u)

print("finished")
print(f"row sum {row_sums}")
print(f"col sum {col_sums}")