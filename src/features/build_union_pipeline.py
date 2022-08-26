import click
import joblib as jb
import mlflow
import pandas as pd
from scipy.sparse import hstack, load_npz, save_npz
from sklearn.pipeline import FeatureUnion

from src.config.config import create_config
from src.models.utils_mlflow import STAGE_STAGING, save_sklearn_model


@click.command()
@click.argument("input_data_sites", type=click.Path(exists=True))
@click.argument("input_data_pref", type=click.Path(exists=True))
@click.argument("input_data_features", type=click.Path(exists=True))
@click.argument("input_transformer_sites", type=click.Path(exists=True))
@click.argument("input_transformer_prefs", type=click.Path(exists=True))
@click.argument("input_transormer_features", type=click.Path(exists=True))
@click.argument("output_data", type=click.Path())
@click.argument("output_transformer", type=click.Path())
@click.option("--save-mlflow", is_flag=True, default=False)
def build_union_pipeline(
    input_data_sites: str,
    input_data_pref: str,
    input_data_features: str,
    input_transformer_sites: str,
    input_transformer_prefs: str,
    input_transormer_features: str,
    output_data: str,
    output_transformer: str,
    save_mlflow: bool,
):
    """Creates union pipeline

    Args:
        input_data_sites (str): path to the dataset w sites (npz)
        input_data_pref (str): path to the dataset w preferences (csv)
        input_data_features (str): path to the dataset wadditional features (npz)
        input_transformer_sites (str): path to the file w transformer sites (pkl)
        input_transformer_prefs (str): path to the file w transformaer preferences (pkl)
        input_transormer_features (str): path to the file w transformer features (pkl)
        output_data (str): path to the output data w all features (npz)
        output_transformer (str): path to the output transformer (pkl)
        save_mlflow (bool): don't save mlflow model
    """
    cfg = create_config()

    # concat train dataset
    X_train_sites = load_npz(input_data_sites)
    X_train_preferences = pd.read_csv(input_data_pref, index_col=0)
    X_train_features = load_npz(input_data_features)
    X_train = hstack([X_train_sites, X_train_preferences, X_train_features])
    save_npz(output_data, X_train)

    vectorizer_pipeline = jb.load(input_transformer_sites)
    preferences_attributes_pipeline = jb.load(input_transformer_prefs)
    features_pipeline = jb.load(input_transormer_features)

    # concat features transformer pipeline
    full_pipeline = FeatureUnion(
        transformer_list=[
            ("vectorizer", vectorizer_pipeline),
            ("preferences", preferences_attributes_pipeline),
            ("features", features_pipeline),
        ]
    )

    jb.dump(full_pipeline, output_transformer)

    # dump model in mlflow as a new staging transformer
    if not save_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        save_sklearn_model(
            model=full_pipeline,
            name=cfg.mlflow.transformer_model_name,
            experiment_name=cfg.mlflow.transformer_experiment_name,
            stage=STAGE_STAGING,
        )


if __name__ == "__main__":
    build_union_pipeline()
