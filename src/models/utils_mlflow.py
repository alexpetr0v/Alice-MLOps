from typing import Final, Optional, Union

import mlflow
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.models.model import ModelInfo
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

STAGE_STAGING: Final[str] = "Staging"
STAGE_PRODUCTION: Final[str] = "Production"
STAGE_NONE: Final[str] = "None"


def get_sklearn_model_version(
    model_name: str, model_stage: Optional[str] = None
) -> ModelVersion:
    """Finds the latest version of the sklearn mlflow server model

    Args:
        model_name (str): mlflow model name
        model_stage (Optional[str], optional):  model stage name (None - the search for Production; then Staging)

    Returns:
        Optional[ModelVersion]: mlflow model version
    """
    mlf_client = mlflow.tracking.MlflowClient()

    if not model_stage:
        request_result = mlf_client.get_latest_versions(
            model_name, stages=[STAGE_PRODUCTION]
        )

        if len(request_result) == 0:
            request_result = mlf_client.get_latest_versions(
                model_name, stages=[STAGE_STAGING]
            )

    else:
        request_result = mlf_client.get_latest_versions(
            model_name, stages=[model_stage]
        )

    if len(request_result) > 0:
        return request_result[0]
    else:
        return None


def save_sklearn_model(
    model: Union[Pipeline, LogisticRegression],
    name: str,
    stage: str,
    experiment_name: str,
) -> ModelInfo:
    """Saves new final model in mlflow in the selected experiment.
    Changes the recently added stage of the model to the selected one.
    The model is saved in a special mlflow run

    Args:
        model (Union[Pipeline, LogisticRegression]): model
        name (str): model name
        stage (str): current stage
        experiment_name (str): experiment name
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(model, name, registered_model_name=name)
        assign_model_stage(model_info, stage)
    return model_info


def assign_model_stage(model_info: ModelInfo, stage: str = STAGE_STAGING) -> None:
    """Modifies the model stage and archives the current model with the selected stage

    Args:
        model_info (ModelInfo): model information
        stage (str, optional): current stage
    """
    mlf_client = mlflow.tracking.MlflowClient()
    model = mlf_client.search_model_versions(f"run_id='{model_info.run_id}'")[0]
    mlf_client.transition_model_version_stage(
        name=model.name,
        version=model.version,
        stage=stage,
        archive_existing_versions=True,
    )
