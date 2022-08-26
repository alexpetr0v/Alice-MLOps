from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.config import DataConfig


class DataPreparator(BaseEstimator, TransformerMixin):
    """
    Prepares CountVectorizer friendly list of strings with sites IDs
    """

    def __init__(self, data_cfg: DataConfig):
        self.sites = data_cfg.sites
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> "DataPreparator":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> List[str]:
        X = X[self.sites].fillna(0).astype(int).values.tolist()
        return [" ".join(map(str, filter(lambda x: x != 0, row))) for row in X]
