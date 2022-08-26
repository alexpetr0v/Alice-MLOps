import pickle

import click
import joblib as jb
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import create_config
from src.features.build_preference_features import PreferencesAttributesAdder


@click.command()
@click.argument("input_dict_sites", type=click.Path(exists=True))
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_pref_data", type=click.Path())
@click.argument("output_transformer", type=click.Path())
def build_preferences_pipeline(
    input_dict_sites: str,
    input_data: str,
    output_pref_data: str,
    output_transformer: str,
):
    """Creates preferences pipeline

    Args:
        input_dict_sites (str): path to the dict with sites (pkl)
        input_data (str): path to the input train dataset (csv)
        output_pref_data (str): path to the train output file with user preferences (csv)
        output_transformer (str): path to the exported transformer (pkl)
    """
    with open(input_dict_sites, "rb") as dict_with_sites:
        dict_sites = pickle.load(dict_with_sites)
    dict_sites_reverse = {name: site for (site, name) in dict_sites.items()}

    cfg = create_config()
    preferences_attributes_pipeline = Pipeline(
        [
            (
                "adder",
                PreferencesAttributesAdder(cfg.data.sites, dict_sites_reverse, cfg.eda),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    df_train = pd.read_csv(
        input_data, index_col=cfg.data.id, parse_dates=cfg.data.times
    )
    y = df_train[cfg.data.target]

    train_preferences = preferences_attributes_pipeline.fit_transform(df_train, y)

    X_train_preference = pd.DataFrame(
        data=train_preferences, columns=["target_pref", "users_pref"]
    )
    X_train_preference.to_csv(output_pref_data)

    jb.dump(preferences_attributes_pipeline, output_transformer)


if __name__ == "__main__":
    build_preferences_pipeline()
