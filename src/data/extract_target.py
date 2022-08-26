import click
import pandas as pd

from src.config import create_config


@click.command()
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_target", type=click.Path())
def extract_target(input_data: str, output_target: str):
    """The method extracts the target column from the original dataset

    Args:
        input_data (str): path to the original dataset
        output_target (str): path to the target
    """
    cfg = create_config()
    df_train = pd.read_csv(input_data)
    y = df_train[cfg.data.target]
    y.to_csv(output_target)


if __name__ == "__main__":
    extract_target()
