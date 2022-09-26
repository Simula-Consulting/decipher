import mlflow
import tensorflow as tf
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from matfact.data_generation import Dataset
from matfact.experiments import train_and_log
from matfact.settings import BASE_PATH, DATASET_PATH


def get_objective(data: Dataset, **hyperparams):
    """Simple train-test based search."""

    X_train, X_test, M_train, M_test = data.get_split_X_M()

    @use_named_args(space)
    def objective(**search_hyperparams):
        hyperparams.update(search_hyperparams)
        output = train_and_log(
            X_train,
            X_test,
            nested=True,
            dict_to_log=data.prefixed_metadata(),
            log_loss=False,
            **hyperparams
        )
        return -output["score"]

    return objective


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    mlflow.set_tracking_uri(BASE_PATH / "mlruns")
    space = (
        Real(-5.0, 1, name="lambda1"),
        Real(8, 20, name="lambda2"),
        Real(0.0, 20, name="lambda3"),
    )

    # Load data
    try:
        data = Dataset().load(DATASET_PATH)
    except FileNotFoundError:  # No data loaded
        data = Dataset().generate(1000, 40, 5, 5)

    with mlflow.start_run() as run:
        res_gp = gp_minimize(
            get_objective(data, convolution=False, shift_range=None), space, n_calls=60
        )
