"""Example usage of class based Matfact API.

Simple demo of the [`MatFact`][matfact.model.matfact.MatFact] class.

Note:
    The current implementation always uses SCMF, as it is the most general solver.
"""

import tensorflow as tf

from matfact import settings
from matfact.data_generation.dataset import Dataset
from matfact.model.config import ModelConfig
from matfact.model.matfact import ArgmaxPredictor, MatFact
from matfact.model.predict.dataset_utils import prediction_data

tf.config.set_visible_devices([], "GPU")


def main():
    dataset = Dataset.from_file(settings.DATASET_PATH)

    X_train, X_test, *_ = dataset.get_split_X_M()
    matfact = MatFact(ModelConfig(), predictor=ArgmaxPredictor())

    matfact.fit(X_train)
    X_test_masked, time_steps, true_states = prediction_data(X_test)

    print(matfact.predict_probabilities(X_test_masked, time_steps))
    print(true_states)
    print(matfact.predict(X_test_masked, time_steps))


if __name__ == "__main__":
    main()
