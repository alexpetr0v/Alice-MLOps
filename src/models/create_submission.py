from typing import List

import click
import pandas as pd
import requests
from pydantic import parse_obj_as

from src.api.predicted_response import PredictedResponse
from src.config import create_config


@click.command()
@click.argument("test_df", type=click.Path(exists=True))
@click.argument("output_submission_file", type=click.Path())
def create_submission(test_df: str, output_submission_file: str):
    """Creates a submission file using predefined parameters of the best model

    Args:
        test_df (str): path to the test dataset (csv)
        output_submission_file (str): path to the submisstion file (csv)
    """
    cfg = create_config()

    test_df = pd.read_csv(test_df, parse_dates=cfg.data.times)
    request_body = test_df.to_json(orient="records")
    response = requests.post(
        cfg.web_service.url + cfg.web_service.predict_proba_method, data=request_body
    )
    predictions = parse_obj_as(List[PredictedResponse], response.json())

    id = "session_id"
    target = "target"
    with open(output_submission_file, "w", encoding="utf-8") as f:
        f.write("{id},{target}\n")
        f.writelines(
            list(
                f"{session.session_id},{session.class_1_proba}\n"
                for session in predictions
            )
        )


if __name__ == "__main__":
    create_submission()
