import click
import joblib as jb
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.config import create_config
from src.features.build_vectorizer import DataPreparator


@click.command()
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_vect", type=click.Path())
@click.argument("output_transformer", type=click.Path())
def build_vectorizer_pipeline(
    input_data: str, output_vect: str, output_transformer: str
):
    """Creates vectorizer pipeline

    Args:
        input_data (str): path to the input dataset (csv)
        output_vect (str): path to the output vect file (npz)
        output_transformer (str): path to the exported transformer (pkl)
    """

    cfg = create_config()
    vectorizer_pipeline = Pipeline(
        [
            ("preparator", DataPreparator(cfg.data)),
            (
                "vectorizer",
                TfidfVectorizer(
                    token_pattern=r"(?u)[1-9]\w*\b",
                    ngram_range=(1, 3),
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    df_train = pd.read_csv(
        input_data, index_col=cfg.data.id, parse_dates=cfg.data.times
    )
    vectorizer_pipeline.fit(df_train)
    X_train_sites = vectorizer_pipeline.transform(df_train)
    save_npz(output_vect, X_train_sites)

    jb.dump(vectorizer_pipeline, output_transformer)


if __name__ == "__main__":
    build_vectorizer_pipeline()
