from typing import Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config.config import EDAConfig


class PreferencesAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Add topics of preferences from Alice and users
    """

    def __init__(
        self, sites: List[str], dict_sites: Dict[str, str], eda_cfg: EDAConfig
    ):
        self.sites = sites
        self.dict_sites = dict_sites
        self.eda_cfg = eda_cfg
        self.alice_sites = set()
        self.users_sites = set()
        super().__init__()

    def fit(self, X: pd.DataFrame, y=None) -> "PreferencesAttributesAdder":
        alice_topics = {
            "cinema",
            "film",
            "media",
            "movie",
            "mtv",
            "music",
            "radio",
            "stream",
            "tv",
            "video",
            "yt3",
            "youwatch",
            "ytimg",
            "youtube",
        }
        alice_topics = set(self.eda_cfg.alice_topics)

        X = X[self.sites].applymap(self.dict_sites.get, na_action="ignore")
        # get the number of Alice sites and people users
        alice_sites_count = X[y == 1].melt().value.value_counts()
        users_sites_count = X[y == 0].melt().value.value_counts()
        # get the best different sites for Alice and users
        self.users_sites = set(users_sites_count.head(50).index).difference(
            alice_sites_count.index
        )
        alice_sites_set = set(alice_sites_count.head(40).index).difference(
            users_sites_count.head(250).index
        )

        self.alice_sites = alice_topics.union(
            [
                site
                for site in alice_sites_set
                if all([at not in site for at in alice_topics])
            ]
        )

        return self

    def transform(self, X):
        X = X[self.sites].applymap(self.dict_sites.get, na_action="ignore")
        alice_preferences = X.apply(
            self.check_preference, axis=1, args=(self.alice_sites,)
        )
        users_preferences = X.apply(
            self.check_preference, axis=1, args=(self.users_sites,)
        )

        return np.c_[alice_preferences, users_preferences]

    def check_preference(self, row: pd.Series, list_of_preferences: Set[str]) -> int:
        """Check pref sites in user sessions

        Args:
            row (pd.Series): visited sites
            list_of_preferences (Set[str]): pref sites

        Returns:
            int: return 1 if pref site is found, otherwise - 0
        """
        for site in row:
            if isinstance(site, str):
                for preference in list_of_preferences:
                    if preference in site:
                        return 1
        return 0
