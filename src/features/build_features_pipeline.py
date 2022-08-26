import click
import joblib as jb
import pandas as pd
from scipy.sparse import save_npz
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import create_config
from src.features.build_categorical_features import CategoricalAttributesAdder
from src.features.build_numeric_features import NumericAttributesAdder


@click.command()
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_features_data", type=click.Path())
@click.argument("output_transformer", type=click.Path())
def build_features_pipeline(
    input_data: str, output_features_data: str, output_transformer: str
):
    """Creates features pipeline

    Args:
        input_data (str): path to the input train dataset (csv)
        output_features_data (str): path to the output train file with features (npz)
        output_transformer (str): path to the exported transformer (pkl)
    """
    cfg = create_config()
    categorical_attributes_pipeline = Pipeline(
        [
            ("adder", CategoricalAttributesAdder(cfg.data)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_attributes_pipeline = Pipeline(
        [
            ("adder", NumericAttributesAdder(cfg.data, cfg.eda)),
            ("scaler", StandardScaler()),
        ]
    )

    features_pipeline = FeatureUnion(
        transformer_list=[
            ("categorical_features", categorical_attributes_pipeline),
            ("numeric_features", numerical_attributes_pipeline),
        ]
    )

    df_train = pd.read_csv(
        input_data, index_col=cfg.data.id, parse_dates=cfg.data.times
    )
    X_train_features = features_pipeline.fit_transform(df_train)

    save_npz(output_features_data, X_train_features)

    jb.dump(features_pipeline, output_transformer)


if __name__ == "__main__":
    build_features_pipeline()
