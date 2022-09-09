from experiments.algorithms.utils import reconstruction_mse
from experiments.main import model_factory
from experiments.algorithms.optimization import matrix_completion
from experiments.algorithms.risk_prediction import predict_proba
from experiments.simulation import prediction_data, data_weights
from experiments.plotting.diagnostic import (plot_confusion, plot_basis, plot_coefs, 
                                 plot_train_loss, plot_roc_curve)
from data_generation.main import Dataset
import mlflow
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score
import tensorflow as tf
from itertools import product

BASE_PATH = "./"

def experiment(
    hyperparams,
    optimization_params,
    enable_shift: bool = False,
    enable_weighting: bool = False,
    enable_convolution: bool = False,
    mlflow_tags: dict = None,
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
    dataset = Dataset().load(f"{BASE_PATH}/datasets")
    X_train, X_test, M_train, M_test = dataset.get_split_X_M()

    # Simulate data for a prediction task by selecting the last data point in each 
    # sample vetor as the prediction target
    X_test_masked, t_pred, x_true = prediction_data(X_test, "last_observed")

    mlflow.start_run()
    mlflow.set_tags(mlflow_tags)
    mlflow.log_params(hyperparams)
    mlflow.log_params(optimization_params)
    mlflow.log_params(dataset.prefixed_metadata())

    # Generate the model
    model_name, model = model_factory(X_train,
        shift_range=np.arange(-12, 13) if enable_shift else np.array([]),
        convolution=enable_convolution,
        weights=data_weights(X_train) if enable_weighting else None,
        **hyperparams)

    mlflow.log_param("model_name", model_name)

    #### Training and testing ####
    # Train the model (i.e. perform the matrix completion)
    extra_metrics = (("recMSE", lambda model: reconstruction_mse(M_train, X_train, model.M)),)
    results = matrix_completion(model, X_train, extra_metrics=extra_metrics, **optimization_params)

    # Predict the risk over the test set using the results from matrix completion as 
    # input parameters to the prediction algorithm 
    p_pred = predict_proba(X_test_masked, results["M"], t_pred, results["theta_mle"])
    # Estimate the mostl likely prediction result from the probabilities 
    x_pred = 1.0 + np.argmax(p_pred, axis=1)

    # Log some metrics
    mlflow.log_metric("matthew_score", matthews_corrcoef(x_true, x_pred), step=results["epochs"][-1])
    mlflow.log_metric("accuracy", accuracy_score(x_true, x_pred), step=results["epochs"][-1])
    for epoch, loss in zip(results["epochs"], results["loss_values"]):
        mlflow.log_metric("loss", loss, step=epoch)
    
    for metric,_ in extra_metrics:
        for epoch, metric_value in zip(results["epochs"], results[metric]):
            mlflow.log_metric(metric, metric_value, step=epoch)

    mlflow.log_metric("norm_difference", np.linalg.norm(results["M"] - M_train))


    ## Plotting ##
    figure_path = f"{BASE_PATH}/results/figures"
    plot_coefs(results["U"], figure_path)
    plot_basis(results["V"], figure_path)
    plot_confusion(x_true, x_pred, figure_path)
    plot_roc_curve(x_true, p_pred, figure_path)
    mlflow.log_artifacts(figure_path)

    mlflow.end_run()

def main():
    USE_GPU = False
    if not USE_GPU:
        tf.config.set_visible_devices([], 'GPU')
    
    mlflow_tags = {
        "Developer": "Thorvald M. Ballestad",
        "GPU": USE_GPU,
        "Notes": "tf.function commented out"
    }
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
        experiment(hyperparams, optimization_params, shift, weight, convolve, mlflow_tags=mlflow_tags)


if __name__ == "__main__":
    main()