from typing import Dict

import mlflow
import numpy as np
import pandas as pd
from mlflow.models.model import ModelInfo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import create_config
from src.models.utils_mlflow import get_sklearn_model_version


def train_model(
    X: pd.DataFrame, y: np.ndarray, parameters: Dict[str, object]
) -> ModelInfo:
    """CV model - train model

    Args:
        X (pd.DataFrame): train dataset
        y (np.ndarray): target
        parameters (Dict[str, str]): model parameters

    Returns:
        information about the mlflow model for the newly created model
    """
    cfg = create_config()
    mlflow.set_experiment(cfg.mlflow.lr_experiment_name)
    mlf_client = mlflow.tracking.MlflowClient()

    with mlflow.start_run():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=21)
        lr = LogisticRegression(**parameters)
        metrics = "roc_auc"
        cv_scores = cross_validate(
            lr, X, y, cv=cv, return_train_score=True, scoring=metrics
        )

        # mean metric values
        scores = {
            metric_: np.mean(val)
            for (metric_, val) in cv_scores.items()
            if any(map(metric_.endswith, metrics))
        }

        # fit
        lr = LogisticRegression(**parameters)
        lr.fit(X, y)

        # add model in mlflow
        model_info = mlflow.sklearn.log_model(lr, cfg.mlflow.lr_model_name)
        model_version = mlflow.register_model(
            model_info.model_uri, cfg.mlflow.lr_model_name
        )
        mlflow.log_metrics(scores)
        mlflow.log_params(parameters)

        # transformer version
        transformer_version = get_sklearn_model_version(
            cfg.mlflow.transformer_model_name
        )
        mlf_client.set_model_version_tag(
            model_version.name,
            model_version.version,
            cfg.mlflow.transformer_tag,
            {"name": transformer_version.name, "version": transformer_version.version},
        )

    return model_info
