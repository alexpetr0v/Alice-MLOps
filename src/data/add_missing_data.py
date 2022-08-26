import datetime

import click
import numpy as np
import pandas as pd

from src.config import create_config


@click.command()
@click.argument("input_data", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
def add_missing_data(input_data: str, output: str):
    """Copied data from week 48 to week 49 from the test dataset

    Args:
        input_data (str): path to the train data
        output (str): path to the train data with missed week 49
    """
    cfg = create_config()

    df_train = pd.read_csv(
        input_data, index_col=cfg.data.id, parse_dates=cfg.data.times
    )
    # copy missed week from previous week
    copied_df = df_train[
        df_train[cfg.data.times[0]].dt.isocalendar().week == cfg.eda.missed_week - 1
    ].copy()
    copied_df[cfg.data.times] = copied_df[cfg.data.times].applymap(
        lambda x: x + datetime.timedelta(days=7)
    )
    # reindex the new rows starting from the maximum index in the existing data frame
    copied_df.reset_index(inplace=True)
    copied_df[cfg.data.id] = np.arange(
        df_train.index.max() + 1, df_train.index.max() + 1 + copied_df.shape[0]
    )
    copied_df.set_index(cfg.data.id, inplace=True)
    df_train = pd.concat([df_train, copied_df], ignore_index=False)
    df_train.to_csv(output)


if __name__ == "__main__":
    add_missing_data()
