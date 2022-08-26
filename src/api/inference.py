from typing import List

import mlflow
import pandas as pd
import yaml
from cachetools import TTLCache, cached
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from mlflow.entities.model_registry.model_version import ModelVersion
from sklearn.pipeline import Pipeline
from yaml.loader import SafeLoader

from src.api.predicted_response import PredictedResponse
from src.api.session_information import SessionInformation
from src.config.config import create_config
from src.models import utils_mlflow

app = FastAPI()
_transformer_version: ModelVersion = None
_model_version: ModelVersion = None


@app.get("/modelversion")
async def get_model_version() -> FileResponse:
    """Provides information about currently selected transformer and prediction models.
    Contains the etag in the response header and can be used as an external dependency in DVC

    Returns:
        FileResponse: mlflow paths to the model information file
    """
    get_pipeline()
    model_info_path = "modelversion.txt"
    model_info_tag = f"Transformer: {_transformer_version._source}, Predictor: {_model_version._source}"
    # dumping the model information file
    with open(model_info_path, "w", encoding="utf-8") as f:
        f.write(model_info_tag)

    response = FileResponse(model_info_path)
    response.headers["etag"] = model_info_tag

    return response


@app.post("/predictproba")
async def predict_proba(
    session_info: List[SessionInformation],
) -> List[PredictedResponse]:
    """The method makes a prediction for the user session using the current model from the mlflow server

    Args:
        session_info (List[SessionInformation]): list w sessions

    Returns:
        List[PredictedResponse]: list w predictions
    """
    cfg = create_config()
    if len(session_info) > cfg.web_service.request_max_items:
        raise HTTPException(400, "Too many elements to predict")

    df = pd.DataFrame(list(dict(session) for session in session_info)).set_index(
        cfg.data.id
    )
    model = get_pipeline()
    prediction = model.predict_proba(df)
    result = [
        PredictedResponse.parse_obj(
            {
                "session_id": df.index.values[i],
                "class_0_proba": prediction[i][0],
                "class_1_proba": prediction[i][1],
            }
        )
        for i in range(len(prediction))
    ]

    return result


@cached(cache=TTLCache(ttl=create_config().web_service.cache_timeout, maxsize=10))
def get_pipeline() -> Pipeline:
    """Get a complete pipeline for processing raw data into predictions

    Returns:
        Pipeline: sklearn Pipeline
    """
    global _transformer_version
    global _model_version
    cfg = create_config()

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlf_client = mlflow.tracking.MlflowClient()

    _model_version = utils_mlflow.get_sklearn_model_version(cfg.mlflow.lr_model_name)
    transformer_tag = yaml.load(
        _model_version.tags[cfg.mlflow.transformer_tag], SafeLoader
    )
    _transformer_version = mlf_client.get_model_version(**transformer_tag)
    transformer = mlflow.sklearn.load_model(_transformer_version.source)
    predictor = mlflow.sklearn.load_model(_model_version._source)
    pipeline = Pipeline((("transformer", transformer), ("predictor", predictor)))

    return pipeline
