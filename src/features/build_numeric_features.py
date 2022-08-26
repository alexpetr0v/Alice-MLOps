import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.config import DataConfig, EDAConfig


class NumericAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Add new numeric features
    """

    def __init__(self, data_cfg: DataConfig, eda_cfg: EDAConfig):
        self.sites = data_cfg.sites
        self.times = data_cfg.times
        self.eda_cfg = eda_cfg
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> "NumericAttributesAdder":
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        sites_count = len(self.sites) - np.sum(X[self.sites].isna(), axis=1)
        # numeric_only=False is a workaround to make pd.Dataframe.sum() work w/ NaT values
        session_time = (
            (
                X[self.times].max(axis=1, numeric_only=False)
                - X[self.times].min(axis=1, numeric_only=False)
            ).dt.seconds
        ) ** 0.1

        time_per_site = (session_time / sites_count) ** 0.1
        alice_time = X[self.times[0]].apply(self.check_alice_time)

        is_morning = (X[self.times[0]].dt.hour <= 11).astype(int)
        is_evening = (X[self.times[0]].dt.time > datetime.time(18, 30)).astype(int)

        return np.c_[session_time, time_per_site, alice_time, is_morning, is_evening]

    def check_alice_time(self, dt: pd.Timestamp) -> int:
        """
        The method returns 1 if the time refers to Alice's standard working hours, otherwise - 0
        """
        for start, end in self.eda_cfg.alice_timetable[dt.day_name()]:
            if dt.time() >= start and dt.time() <= end:
                return 1
        return 0
