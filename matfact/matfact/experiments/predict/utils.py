import numpy as np 
import tensorflow as tf 


# TODO: Should handle arbitrary classes, not just binary 
class BinaryDeltaLoss(tf.keras.losses.Loss):
	"Loss function for training with delta scores on a binary class problem."

	def call(self, y_true, y_pred):

		y_pred = tf.squeeze(tf.convert_to_tensor(y_pred))
		y_true = tf.squeeze(tf.cast(y_true, y_pred.dtype))
		
		return y_true * (1 - 2 * y_pred) + (1 - y_true) * (2 * y_pred - 1)


# TODO: Should handle arbitrary classes, not just binary 
def binary_delta_score(y_true, p_pred):
	"""Compute the individual delta scores for model predictions on a 
	binary (two-class) problem.

	Args:
		y_true (ndarray): Binary indicator vector of class memberships per data sample.
		p_pred (ndarray): Vector model predictions for the target (positive) class.

	Returns:	
		Vector of delta scores corresponding to model predictions.  
	"""

	y_true = y_true.astype(int)
	p_pred = np.transpose(np.vstack([1.0 - p_pred, p_pred]))

	return p_pred[range(y_true.size), 1 - y_true] - p_pred[range(y_true.size), y_true] 


# TODO: Should handle arbitrary classes, not just binary 
class DeltaConvergenceMonitor:
	"""Evaluate if an algorithm trained with delta loss has converged. 
	Evalautes Eq. (9) in manuscript "Adaptive sample weighting for treating 
	imbalanced data in cervical cancer risk prediction".

	Args:
		delta (ndarray): Vector of delta scores.
		w_delta: (ndarray): Initial estimate for sample weights. 
		sample_weights_fn (callable): Function to re-estimate delta scores. 
			Corresponds to Eq. (8) in manuscript. 
		tau (optional): Convergence threshold. 
	"""

	def __init__(self, delta, w_delta, sample_weights_fn, tau=0.0001):

		self.delta = delta
		self.w_tilde = w_delta
		self.sample_weights_fn = sample_weights_fn

		self.tau = tau
		self.rates = []

	def _binary_delta_score(self, y_true, p_pred):

		# NOTE: Assume p_pred are probabilities for the target class 
		# and that 1 - p_pred are probabilities for the alternative class.
		y_true = y_true.astype(int)
		p_pred = np.transpose(np.vstack([1.0 - p_pred, p_pred]))

		return p_pred[range(y_true.size), 1 - y_true] - p_pred[range(y_true.size), y_true] 

	def update_iter(self, y_true, p_hat, alpha, update_i):
		"""Update parameters for assessing convergence.

		Args:
			y_true (ndarray): Indicator vector of ground truths.  
			p_hat (ndarray): Vector of predicted target class probailities. 
			alpha (float): Parameter for update speed (see Eq. (i) in manuscript).
			update_i (int): Iteration number in the training process (i.e., epoch number).
		"""
	
		delta = binary_delta_score(y_true, p_hat)

		self.w_delta = self.sample_weights_fn(y_true, p_hat, update_i, alpha)
		self.w_tilde = alpha * self.w_tilde + (1 - alpha) * self.w_delta

		self.eta = np.linalg.norm(delta - self.delta) ** 2 / y_true.size 
		self.rates.append(self.eta)

		# NOTE: Update delta only after compute eta 
		self.delta = delta
			
	def should_stop(self):

		return self.eta < self.tau
