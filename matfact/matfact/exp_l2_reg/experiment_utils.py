import logging
import pathlib

import numpy as np
import pandas as pd

from urllib.parse import urlparse

from mlflow.entities import ViewType

from matfact.data_generation.dataset import Dataset, DGD, HMM
from .plot_utils import NORMAL, INVERTED


def matrix_info(m):
    return [
        (int(value), int(count))
        for value, count in np.vstack(np.unique(np.round(m), return_counts=True)).T
    ]


def log_dataset_info(dataset, inverted=False):
    preffix = INVERTED if inverted else NORMAL
    logging.info(f"{preffix} dataset X histogram: {matrix_info(dataset.X)}")
    logging.info(f"{preffix} dataset M histogram: {matrix_info(dataset.M)}")
    logging.info(
        f"{preffix} dataset generation method: {dataset.metadata['generation_method']}"
    )


def invert_domain(m):
    # Inverts matrix label distributions
    return m.max() - (m - m.min())


def invert_domain_excluding_value(m, exclude):
    # Inverts matrix label distributions excluding a value
    inv_m = np.zeros(m.shape)
    inv_m[m != exclude] = invert_domain(m[m != exclude])
    return inv_m


def invert_dataset(dataset):
    # Inverts dataset label distributions
    log_dataset_info(dataset)
    if dataset.metadata["generation_method"] == DGD:
        inv_M = invert_domain(dataset.M.copy())
    elif dataset.metadata["generation_method"] == HMM:
        inv_M = invert_domain_excluding_value(dataset.M.copy(), 0)
    inv_X = invert_domain_excluding_value(dataset.X.copy(), 0)
    inv_metadata = dataset.metadata.copy()
    inv_metadata["observation_probabilities"] = [0.01, 0.04, 0.12, 0.08, 0.03]
    inv_dataset = Dataset(inv_X, inv_M, inv_metadata)
    log_dataset_info(inv_dataset, inverted=True)
    return inv_dataset


def fetch_experiment_logs(client, experiment_id):
    # Fetch experimet run logs
    experiment_runs = client.search_runs(
        experiment_ids=experiment_id,
        run_view_type=ViewType.ALL,
        order_by=["metric.matthew_score ASC"],
    )

    # Loop through experiment runs and retrieve metrics and artifacts
    artifacts_logs, run_logs = [], []
    for run in experiment_runs:
        # Fetch run_id and artifacts path
        run_id = run.info.run_id
        # Fetch flat logs
        run_log = {"run_id": run_id}
        run_log.update(run.data.params)
        run_log.update(run.data.metrics)
        run_logs.append(run_log)
        # Fetch logged artifacts
        artifacts_log = {"run_id": run_id}
        artifacts_log.update(run.data.params)
        artifacts_path = pathlib.Path(urlparse(run.info.artifact_uri).path)
        for artifact in client.list_artifacts(run_id):
            artifact_path = artifacts_path / artifact.path
            artifacts_log[artifact_path.name] = artifact_path
        client.set_terminated(run_id)
        artifacts_logs.append(artifacts_log)
    # Build dataframe with list of logs and return
    logs_df = pd.DataFrame(run_logs).astype({"lambda1": float, "lambda2": float})
    artifacts_df = pd.DataFrame(artifacts_logs).astype(
        {"lambda1": float, "lambda2": float}
    )
    return logs_df, artifacts_df
