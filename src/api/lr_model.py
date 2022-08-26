import mlflow
import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression

from src.config.config import create_config


class LogisticRegressionModel:
    """Loads a trained logistic regression model from mlflow
    and provides access to its prediction methods
    """

    def __init__(self, model_name: str, model_stage: str):
        cfg = create_config()
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlf_client = mlflow.tracking.MlflowClient()
        lr_model = mlf_client.get_latest_versions(model_name, stages=[model_stage])[0]
        lr = mlflow.sklearn.load_model(f"models:/{lr_model.name}/{lr_model.version}")
        self._model: LogisticRegression = lr

    def predict_proba(self, X: spmatrix) -> np.ndarray:
        return self._model.predict_proba(X)
