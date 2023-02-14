"""Example usage of class based Matfact API.

Simple demo of the [`MatFact`][matfact.model.matfact.MatFact] class.
"""


import contextlib

import tensorflow as tf

from decipher.data_generation.dataset import Dataset
from decipher.matfact.model.config import ModelConfig
from decipher.matfact.model.matfact import ArgmaxPredictor, MatFact
from decipher.matfact.model.predict.dataset_utils import prediction_data
from decipher.matfact.settings import settings

# Disabling the GPU makes everything faster.
# If tf is already initialized, we cannot modified visible devices, in which
# case we just proceed.
with contextlib.suppress(RuntimeError):
    tf.config.set_visible_devices([], "GPU")


def main():
    dataset = Dataset.from_file(settings.paths.dataset)

    X_train, X_test, *_ = dataset.get_split_X_M()
    matfact = MatFact(ModelConfig(), predictor=ArgmaxPredictor())

    matfact.fit(X_train)
    X_test_masked, time_steps, true_states = prediction_data(X_test)

    print(matfact.predict_probabilities(X_test_masked, time_steps))
    print(true_states)
    print(matfact.predict(X_test_masked, time_steps))


if __name__ == "__main__":
    main()
