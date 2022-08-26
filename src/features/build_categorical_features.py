import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.config import DataConfig


class CategoricalAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Add new categorical features
    """

    def __init__(self, data_cfg: DataConfig):
        self.sites = data_cfg.sites
        self.times = data_cfg.times
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalAttributesAdder":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        dayofweek = X[self.times[0]].dt.dayofweek
        sites_count = len(self.sites) - np.sum(X[self.sites].isna(), axis=1)
        week = X[self.times[0]].dt.isocalendar().week

        return np.c_[dayofweek, sites_count, week]
