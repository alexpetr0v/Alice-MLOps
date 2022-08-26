from typing import List

import click
import mlflow
import numpy as np
import pandas as pd
from mlflow.models.model import ModelInfo
from scipy.sparse import load_npz

from src.config import create_config
from src.models.train_model import train_model
from src.models.utils_mlflow import (
    STAGE_NONE,
    STAGE_STAGING,
    assign_model_stage,
    get_sklearn_model_version,
)


@click.command()
@click.argument("X", type=click.Path(exists=True))
@click.argument("y", type=click.Path(exists=True))
@click.argument("output_mlflow_model", type=click.Path())
@click.option("--remove_history", "--ch", is_flag=True)
def optimize_model(X: str, y: str, output_mlflow_model: str, remove_history: bool):
    """Trains multiple models with different hyperparameters and
    calculates training/testing results using cross-validation.
    Pre-selects the best model if there is no Production model
    in mlflow and save in output_mlflow_model

    Args:
        X (str): path to the train dataset (npz)
        y (str): path to the target (csv)
        remove_history (bool): remove runs and model history
        output_mlflow_model (str): path to the file w mlflow models ID (txt)
    """
    cfg = create_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.set_experiment(cfg.mlflow.lr_experiment_name)
    mlf_client = mlflow.tracking.MlflowClient()

    if remove_history:
        remove_experiment_history(experiment, mlf_client)

    # load train data
    X = load_npz(X)
    y = pd.read_csv(y)[cfg.data.target].values

    # CV model w HP
    c_vals = np.linspace(1, 15, 9)
    model_parameters = {
        "n_jobs": 4,
        "multi_class": "ovr",
        "random_state": 21,
        "solver": "lbfgs",
    }
    model_infos: List[ModelInfo] = []
    for c in c_vals:
        model_parameters["C"] = c
        model_info = train_model(X, y, model_parameters)
        model_infos.append(model_info)

    # check if the selected Production or Staging model exist
    # NF: assign the best of the newly trained and save it
    if not get_sklearn_model_version(cfg.mlflow.lr_model_name):
        best_model = None
        best_roc_auc = 0
        for model in model_infos:
            roc_auc = mlf_client.get_metric_history(model.run_id, "test_roc_auc")
            if roc_auc[0].value > best_roc_auc:
                best_roc_auc = roc_auc[0].value
                best_model = model
        assign_model_stage(best_model, STAGE_STAGING)
    # dump of the currently used model
    model_version = get_sklearn_model_version(cfg.mlflow.lr_model_name)
    if model_version:
        with open(output_mlflow_model, "w", encoding="utf-8") as f:
            f.write(model_version.source)


def remove_experiment_history(experiment, mlf_client):
    for run in mlflow.list_run_infos(experiment.experiment_id):
        delete_run = True
        models = mlf_client.search_model_versions(f"run_id='{run.run_id}'")

        for model in models:
            if model.current_stage == STAGE_NONE:
                mlf_client.delete_model_version(model.name, model.version)
            else:
                delete_run = False

        if delete_run:
            mlflow.delete_run(run.run_id)


if __name__ == "__main__":
    optimize_model()
