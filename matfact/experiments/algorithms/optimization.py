from .utils import theta_mle
from .convergence import convergence_monitor
from tqdm import tqdm
 

def matrix_completion(model, X, fname="", epochs_per_val=5, num_epochs=2000, patience=200):
	"Run matrix completion on input matrix X using a factorization model." 
	
	# Results collected from the process 
	output = {
		"convergence_rate": [], 
		"loss_values": [],
		"epochs": [],
		"U": None, 
		"V": None,
		"M": None,
		"s": None,
		"theta_mle": None 
	}
	
	for epoch in tqdm(range(num_epochs)):
		
		model.run_step()
 
		output["epochs"].append(int(epoch))
		output["loss_values"].append(float(model.loss()))

		if epoch == patience:
			monitor = convergence_monitor(model.M)
			
		if epoch % epochs_per_val == 0 and epoch > patience:
			
			if monitor.converged(model.M):
				break

	output["U"] = model.U 
	output["V"] = model.V
	output["M"] = model.M

	if hasattr(model, "s"):
		output["s"] = model.s 

	output["theta_mle"] = theta_mle(X, model.M)

	return output 
