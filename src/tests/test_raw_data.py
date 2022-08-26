import pickle

import great_expectations as ge
import pandas as pd
from great_expectations.dataset import PandasDataset

from src.config import create_config


def test_data_format():
    cfg = create_config()
    df = pd.read_csv(
        "data/raw/train_sessions.csv", index_col=cfg.data.id, parse_dates=cfg.data.times
    )
    df_ge: PandasDataset = ge.from_pandas(df)
    expected_columns = [
        "site1",
        "time1",
        "site2",
        "time2",
        "site3",
        "time3",
        "site4",
        "time4",
        "site5",
        "time5",
        "site6",
        "time6",
        "site7",
        "time7",
        "site8",
        "time8",
        "site9",
        "time9",
        "site10",
        "time10",
        "target",
    ]
    assert (
        df_ge.expect_table_columns_to_match_ordered_list(expected_columns).success
        is True
    ), "train_sessions.csv the columns of the table do not correspond to an ordered list"
    assert (
        df_ge.expect_column_values_to_be_in_set("target", [0, 1]).success is True
    ), "train_sessions.csv the target column contains not only 0 and 1"


def test_data_exists():
    cfg = create_config()
    df = pd.read_csv(
        "data/raw/train_sessions.csv", index_col=cfg.data.id, parse_dates=cfg.data.times
    )
    df_ge: PandasDataset = ge.from_pandas(df)
    assert (
        df_ge.expect_table_row_count_to_be_between(100_000, None).success is True
    ), "the number of rows in the train dataset is too small"


def test_data_types():
    cfg = create_config()

    df_to_check = ["data/raw/train_sessions.csv", "data/raw/test_sessions.csv"]
    for df_file in df_to_check:
        df = pd.read_csv(df_file, index_col=cfg.data.id, parse_dates=cfg.data.times)
        df_ge: PandasDataset = ge.from_pandas(df)

        assert (
            df_ge.expect_column_values_to_not_be_null("site1").success is True
        ), f"{df_file} site1 contains null"
        assert (
            df_ge.expect_column_values_to_not_be_null("time1").success is True
        ), f"{df_file} time1 contains null"
        assert (
            df_ge.expect_column_values_to_be_of_type(cfg.data.sites[0], "int64").success
            is True
        ), f"{cfg.data.sites[0]} in {df_file} is not int64"
        for column_name in cfg.data.sites[1:]:
            assert (
                df_ge.expect_column_values_to_be_of_type(column_name, "float64").success
                is True
            ), f"{column_name} in {df_file} is not float64"
        for column_name in cfg.data.times:
            assert (
                df_ge.expect_column_values_to_be_of_type(
                    column_name, "datetime64"
                ).success
                is True
            ), f"{column_name} in {df_file} is not datetime64"


if __name__ == "__main__":
    test_data_format()
    test_data_types()
    test_data_exists()
