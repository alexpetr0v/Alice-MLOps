import click
import pandas as pd

from src.config import create_config


@click.command()
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output_clean_data", type=click.Path())
def remove_bad_data(input_data: str, output_clean_data: str):
    """Deletes rows for the date 2013-04-12

    Args:
        input_data (str): path to the original dataset
        output_clean_data (str): path to the cleaned file from abnormal value
    """
    cfg = create_config()

    df_train: pd.DataFrame = pd.read_csv(
        input_data, index_col=cfg.data.id, parse_dates=cfg.data.times
    )
    df_train = df_train[
        ~(
            (df_train.target == 1)
            & (df_train[cfg.data.times[0]].dt.date == cfg.eda.abnormal_day)
        )
    ]
    df_train.to_csv(output_clean_data)


if __name__ == "__main__":
    remove_bad_data()
