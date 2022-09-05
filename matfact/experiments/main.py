"""This module demonstrates how to instatiate matrix factorization models
for matrix completion and risk prediction. The example is based on synthetic data 
produced in the `datasets` directory.
"""
import numpy as np 

from plotting.diagnostic import (plot_confusion, plot_basis, plot_coefs, 
                                 plot_train_loss, plot_roc_curve)

from simulation import prediction_data, data_weights, train_test_split

from algorithms import CMF, WCMF, SCMF
from algorithms.optimization import matrix_completion
from algorithms.risk_prediction import predict_proba 

from algorithms.utils import (initialize_basis, 
                              finite_difference_matrix, 
                              laplacian_kernel_matrix)

from mlflow import log_metric, log_param, log_artifacts

BASE_PATH = "/Users/thorvald/Documents/Decipher/decipher/matfact/"  # TODO: make generic
BASE_PATH = "./"


def l2_regularizer(X, rank=5, lambda1=1.0, lambda2=1.0, weights=None, seed=42):
    """Matrix factorization with L2 regularization. Weighted discrepancy term is optional.

    Args:
        X: Sparse (N x T) data matrix used to estimate factor matrices 
        rank: Rank of the factor matrices 
        lambda: Regularization coefficients 
        weights (optional): Weight matrix (N x T) for the discrepancy term. This matrix
            should have zeros in the same entries as X.
    
    Returns:
        A CMF/WCMF object with only L2 regularization.
    """

    # Initialize basic vectors 
    V = initialize_basis(X.shape[1], rank, seed)

    if weights is None:
        return CMF(X, V, lambda1=lambda1, lambda2=lambda2)

    return WCMF(X, V, W=weights, lambda1=lambda1, lambda2=lambda2)


def convolution(X, rank=5, lambda1=1.0, lambda2=1.0, lambda3=1.0, weights=None, seed=42):
    """Matrix factorization with L2 and convolutional regularization. Weighted discrepancy 
    term is optional. The convolutional regularization allows for more local variability in 
    the reconstructed data. 

    Args:
        X: Sparse (N x T) data matrix used to estimate factor matrices 
        rank: Rank of the factor matrices 
        lambda: Regularization coefficients 
        weights (optional): Weight matrix (N x T) for the discrepancy term. This matrix
            should have zeros in the same entries as X.
    
    Returns:
        A CMF/WCMF object with L2 and convolutional regularization.
    """

    # Initialize basic vectors 
    V = initialize_basis(X.shape[1], rank, seed)

    # Construct forward difference (D) and convolutional matrix (K). 
    # It is possible to choose another form for the K matrix. 
    D = finite_difference_matrix(X.shape[1]) 
    K = laplacian_kernel_matrix(X.shape[1])

    if weights is None:
        return CMF(X, V, D=D, K=K, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    return WCMF(X, V, D=D, K=K, W=weights, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)


def shifted(X, rank=5, s_budget=np.arange(-12, 13), lambda1=1.0, lambda2=1.0, lambda3=1.0, 
            weights=None, convolution=False, seed=42):
    """Shifted matrix factorization with L2 and optional convolutional regularization. Weighted discrepancy 
    term is also optional. The shift 
    
    Note that the shifted models (SCMF) are slower than CMF and WCFM.

    Args:
        X: Sparse (N x T) data matrix used to estimate factor matrices 
        rank: Rank of the factor matrices 
        s_budget: The range of possible shifts. 
        lambda: Regularization coefficients 
        weights (optional): Weight matrix (N x T) for the discrepancy term. This matrix
            should have zeros in the same entries as X.
        convolution (bool): If should include convolutional regularisation. 
    
    Returns:
        A CMF/WCMF object with L2 and convolutional regularization.
    """
    
    V = initialize_basis(X.shape[1], rank, seed)

    D, K = None, None 
    if convolution:

        D = finite_difference_matrix(X.shape[1] + 2 * s_budget.size)
        K = laplacian_kernel_matrix(X.shape[1]  + 2 * s_budget.size)
        
    if weights is None:
        return SCMF(X, V, s_budget, D=D, K=K, W=(X != 0).astype(np.float32), lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)

    return SCMF(X, V, s_budget, D=D, K=K, W=weights, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3)


def main():

    # Load some synthetic data from the Gaussian generator 
    X_train, X_test, M_train, M_test = train_test_split(np.load(f"{BASE_PATH}/datasets/X.npy"), 
                                                        np.load(f"{BASE_PATH}/datasets/M.npy"))
    
    # Simulate data for a prediction task by selecting the last data point in each 
    # sample vetor as the prediction target
    X_test_masked, t_pred, x_true = prediction_data(X_test, "last_observed")

    # Examples on how to instantiate a series of models 

    # Only L2 regularization  
    l2mf = l2_regularizer(X_train)

    # Weighted L2 regularization  
    l2mf = l2_regularizer(X_train, weights=data_weights(X_train))

    # Convolutional regularization 
    cmf = convolution(X_train)

    # Weighted discrepancy with convolutional regularization 
    wcmf = convolution(X_train, weights=data_weights(X_train))

    # Shifted L2 regularization  
    sl2mf = shifted(X_train)
    
    # Shifted convolutional regularization  
    scmf = shifted(X_train, convolution=True)

    # Shifted weighted convolutional regularization  
    swcmf = shifted(X_train, convolution=True, weights=data_weights(X_train))

    # Estimate factor matrices U and V such that U @ V.T \approx X 
    # Results from the experiment are stored in the output
    results = matrix_completion(wcmf, X_train)
    log_param("model", "wcmf")

    # Predict the risk over the test set using the results from matrix completion as 
    # input parameters to the prediction algorithm 
    p_pred = predict_proba(X_test_masked, results["M"], t_pred, results["theta_mle"])
    # Estimate the mostl likely prediction result from the probabilities 
    x_pred = 1.0 + np.argmax(p_pred, axis=1)

    # Save useful results 
    np.save(f"{BASE_PATH}/results/data/X_train.npy", X_train)
    np.save(f"{BASE_PATH}/results/data/M_train.npy", M_train)
    np.save(f"{BASE_PATH}/results/data/X_test.npy", X_test)
    np.save(f"{BASE_PATH}/results/data/M_test.npy", M_test)
    np.save(f"{BASE_PATH}/results/data/X_test_masked.npy", X_test_masked)

    np.save(f"{BASE_PATH}/results/data/U.npy", results["U"])
    np.save(f"{BASE_PATH}/results/data/V.npy", results["V"])
    np.save(f"{BASE_PATH}/results/data/M.npy", results["M"])
    np.save(f"{BASE_PATH}/results/data/theta_mle.npy", results["theta_mle"])
    np.save(f"{BASE_PATH}/results/data/epochs.npy", results["epochs"])
    np.save(f"{BASE_PATH}/results/data/loss_values.npy", results["loss_values"])
    np.save(f"{BASE_PATH}/results/data/convergence_rate.npy", results["convergence_rate"])

    np.save(f"{BASE_PATH}/results/data/p_pred.npy", p_pred)
    np.save(f"{BASE_PATH}/results/data/x_pred.npy", x_pred)
    np.save(f"{BASE_PATH}/results/data/x_true.npy", x_true)
    np.save(f"{BASE_PATH}/results/data/t_pred.npy", t_pred)

    # Plotting results 
    plot_coefs(np.load(f"{BASE_PATH}/results/data/U.npy"), f"{BASE_PATH}/results/figures")
    plot_basis(np.load(f"{BASE_PATH}/results/data/V.npy"), f"{BASE_PATH}/results/figures")
    
    plot_train_loss(np.load(f"{BASE_PATH}/results/data/epochs.npy"), 
                     np.load(f"{BASE_PATH}/results/data/loss_values.npy"), f"{BASE_PATH}/results/figures")

    plot_confusion(np.load(f"{BASE_PATH}/results/data/x_true.npy"), 
                   np.load(f"{BASE_PATH}/results/data/x_pred.npy"), f"{BASE_PATH}/results/figures")

    plot_roc_curve(np.load(f"{BASE_PATH}/results/data/x_true.npy"), 
                   np.load(f"{BASE_PATH}/results/data/p_pred.npy"), f"{BASE_PATH}/results/figures")

    log_artifacts(f"{BASE_PATH}/results/figures")

if __name__ == "__main__":
    main()