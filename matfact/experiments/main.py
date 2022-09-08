"""This module demonstrates how to instatiate matrix factorization models
for matrix completion and risk prediction. The example is based on synthetic data 
produced in the `datasets` directory.
"""
from typing import Any
import numpy as np 
import json
from tqdm import tqdm

from plotting.diagnostic import (plot_confusion, plot_basis, plot_coefs, 
                                 plot_train_loss, plot_roc_curve)

from simulation import prediction_data, data_weights, train_test_split

from algorithms import CMF, WCMF, SCMF
from algorithms.optimization import matrix_completion
from algorithms.risk_prediction import predict_proba 

from algorithms.utils import (initialize_basis, 
                              finite_difference_matrix, 
                              laplacian_kernel_matrix,
                              reconstruction_mse)

from sklearn.metrics import matthews_corrcoef, accuracy_score
from itertools import combinations, product
import tensorflow as tf

from mlflow import log_metric, log_param, log_params, log_artifacts, start_run, log_metrics, end_run, set_tag, set_tags

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


def model_factory(X, shift_range: np.ndarray[Any, int], convolution: bool, weights=None, rank: int = 5, seed: int = 42, **kwargs):
    """
    
    TODO: it is possible to move the bool inputs to rather be the absence of values
    for corresponding quantities (non-zoer shift, weights etc)

    TODO: generate a short format model name? Or is it better to simply store the settings?
     """
    
    padding = 2 * shift_range.size

    if convolution:
        D = finite_difference_matrix(X.shape[1] + padding)
        K = laplacian_kernel_matrix(X.shape[1]  + padding)
    else:
        D = K = None  # None will make the objects generate appropriate identity matrices later.
    kwargs.update({"D": D, "K": K})

    V = initialize_basis(X.shape[1], rank, seed)

    short_model_name = "".join(a if cond else b for cond, a, b in [
            (shift_range.size, "s", ""),
            (convolution, "c", "l2"),
            (weights is not None, "w", ""),
        ]) + "mf"

    if shift_range.size:
        weights = (X != 0).astype(np.float32) if weights is None else weights
        return short_model_name, SCMF(X, V, shift_range, W=weights, **kwargs)
    else:
        if weights is not None:
            return short_model_name, WCMF(X, V, weights, **kwargs)
        else:
            return short_model_name, CMF(X, V, **kwargs)

    
def experiment(
    hyperparams,
    optimization_params,
    enable_shift: bool = False,
    enable_weighting: bool = False,
    enable_convolution: bool = False,
):
    """
    hyperparams = {
        rank,
        lambda1,
        lambda2,
    }
    optimization_params = {
        num_epochs,
        epochs_per_val,
        patience,
    }

    TODO: option to plot and log figures
    TODO: option to store and log artifacts like U,V,M,datasets,etc
    TODO: more clearly separate train and predict
    """
    #### Setup and loading ####
    # Load some synthetic data from the Gaussian generator 
    X_train, X_test, M_train, M_test = train_test_split(np.load(f"{BASE_PATH}/datasets/X.npy"), 
                                                        np.load(f"{BASE_PATH}/datasets/M.npy"))
    
    with open(f"{BASE_PATH}/datasets/dataset_metadata.json", "r") as metadata_file:
        dataset_metadata = {f"DATASET_{key}": value for key,value in json.load(metadata_file).items()}

    start_run()
    set_tags({
        "Developer": "Thorvald M. Ballestad",
        "GPU": False,
        "Notes": "tf.function commented out"
    })
    # Probably should not do this, but let us just test
    set_tag("mlflow.note.content", "tf.function is commented out")

    # Simulate data for a prediction task by selecting the last data point in each 
    # sample vetor as the prediction target
    X_test_masked, t_pred, x_true = prediction_data(X_test, "last_observed")

    log_params(hyperparams)
    log_params(optimization_params)
    log_params(dataset_metadata)

    # Generate the model
    model_name, model = model_factory(X_train,
        shift_range=np.arange(-12, 13) if enable_shift else np.array([]),
        convolution=enable_convolution,
        weights=data_weights(X_train) if enable_weighting else None,
        **hyperparams)

    log_param("model_name", model_name)

    #### Training and testing ####
    # Train the model (i.e. perform the matrix completion)
    extra_metrics = (("recMSE", lambda model: reconstruction_mse(M_train, X_train, model.M)),)
    results = matrix_completion(model, extra_metrics=extra_metrics, **optimization_params)

    # Predict the risk over the test set using the results from matrix completion as 
    # input parameters to the prediction algorithm 
    p_pred = predict_proba(X_test_masked, results["M"], t_pred, results["theta_mle"])
    # Estimate the mostl likely prediction result from the probabilities 
    x_pred = 1.0 + np.argmax(p_pred, axis=1)

    # Log some metrics
    log_metric("matthew_score", matthews_corrcoef(x_true, x_pred), step=results["epochs"][-1])
    log_metric("accuracy", accuracy_score(x_true, x_pred), step=results["epochs"][-1])
    for epoch, loss in zip(results["epochs"], results["loss_values"]):
        log_metric("loss", loss, step=epoch)
    
    for metric,_ in extra_metrics:
        for epoch, metric_value in zip(results["epochs"], results[metric]):
            log_metric(metric, metric_value, step=epoch)

    log_metric("norm_difference", np.linalg.norm(results["M"] - M_train))
    end_run()


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')
    # NB! lamabda1, lambda2, lambda3 does *not* correspond directly to 
    # the notation used in the master thesis.
    hyperparams = {
        "rank": 5,
        "lambda1": 10,
        "lambda2": 10,
        "lambda3": 100,
    }
    optimization_params = {
        "num_epochs": 1000,
        "patience": 5,
    }

    for shift, weight, convolve in product([False, True], repeat=3):
        print("Running ", shift, weight, convolve)
        experiment(hyperparams, optimization_params, shift, weight, convolve)