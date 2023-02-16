import pathlib
from urllib.parse import urlparse

import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import ViewType


def fetch_experiment_logs(client: MlflowClient, experiment_id: str):
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
