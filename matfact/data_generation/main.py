"""This module demonstrates how to generate synthetic data developed to 
esemble the screening data used in the DeCipher project.
"""
import os 
import numpy as np
import json

from scipy.stats import betabinom

from masking import simulate_mask
from gaussian_generator import float_matrix, discretise_matrix

BASE_PATH = "/Users/thorvald/Documents/Decipher/decipher/matfact/"  # TODO: make generic
BASE_PATH = "./"

def censoring(X, missing=0):
	"Truncate histories to have patterns similar to the real histories"

	t_cens = betabinom.rvs(n=X.shape[1], a=4.57, b=5.74, size=X.shape[0])
	for i, t_end in enumerate(t_cens):
		X[i, t_end:] = missing

	return X


def produce_dataset(N, T, r, level, memory_length=5, missing=0, 		
					value_range=np.arange(1, 5), theta=2.5, seed=42):
	"""Generate a synthetic dataset resembling the real screening data.
	
	Args:
		N: Number of rows 
		T: Number of columns (time points)
		r: Rank of the ground truth factor matrices  
		level: Level of missing values 
		memory_length: Governs how much influence previous values will 
			have on observing a new value not too far into the future 
		missing: How to indicate a missing value 
		value_range: Possible observed values 
		theta: Confidence parameter 
		seed: Reference value for pseudo-random generator 

	Returns:
		The sparse and original complete data matrices 
	"""

	M = float_matrix(N=N, T=T, r=r, domain=value_range, seed=seed)
	Y = discretise_matrix(M, domain=value_range, theta=theta, seed=seed)

	# Simulate a sparse dataset.
	O = simulate_mask(Y, 
  					  observation_proba=np.array([0.01, 0.03, 0.08, 0.12, 0.04]),
					  memory_length=memory_length,
					  level=level,
					  seed=seed)
	X = O * Y 
	X = censoring(X, missing=missing)

	valid_rows = np.sum(X != 0, axis=1) > 2 
	
	return X[valid_rows].astype(np.float32), M[valid_rows].astype(np.float32)


def main(): 
	
	rank = 5
	n_rows = 1000
	n_columns = 50
	sparsity_level = 6

	X, M = produce_dataset(N=n_rows, T=n_columns, r=rank, level=sparsity_level)

	dataset_metadata = {
		"rank": rank,
		"sparsity_level": sparsity_level,
		"n_rows": n_rows,
		"n_columns": n_columns,
		"generation_method": "DGD",  # Only one method implemented.
	}

	np.save(f"{BASE_PATH}/datasets/X.npy", X)
	np.save(f"{BASE_PATH}/datasets/M.npy", M)
	with open(f"{BASE_PATH}/datasets/dataset_metadata.json", "w") as metadata_file:
		metadata_file.write(json.dumps(dataset_metadata))
	

if __name__ == "__main__":
	main()
