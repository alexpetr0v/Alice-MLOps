import logging
import os
from datetime import date, time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from mergedeep import merge
from pydantic import BaseModel
from yaml import safe_load


class DataConfig(BaseModel):
    id: str
    target: str
    sites: List[str]
    times: List[str]


class EDAConfig(BaseModel):
    missed_week: int
    abnormal_day: date
    alice_timetable: Dict[str, List[Tuple[time, time]]]
    alice_topics: List[str]


class MLflowConfig(BaseModel):
    tracking_uri: str
    lr_model_name: str
    lr_experiment_name: str
    transformer_model_name: str
    transformer_experiment_name: str
    transformer_tag: str


class WebServiceConfig(BaseModel):
    url: str
    predict_proba_method: str
    cache_timeout: int
    cache_size: int
    request_max_items: int


class Config(BaseModel):
    data: DataConfig
    eda: EDAConfig
    mlflow: MLflowConfig
    web_service: WebServiceConfig


_config: Optional[Config] = None


def create_config(config_path="cfg/cfg.yaml") -> "Config":
    """Creates cfg settings

    Args:
        config_path (str): path to the default config file

    Returns:
        Config: cfg
    """
    global _config
    if not _config:
        logging.info("Loading default cfg %s", config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_values = safe_load(f)
        logging.info("Loaded default cfg %s", config_path)

        load_dotenv()
        cfg_addn = os.getenv("alice-TD1tsSPw-py3.9")
        if cfg_addn:
            logging.info("Loading cfg %s", cfg_addn)
            with open(cfg_addn, "r", encoding="utf-8") as f:
                cfg_addon = safe_load(f)
                cfg_values = merge(cfg_values, cfg_addon)
            logging.info("Loaded cfg %s", cfg_addn)
        _config = Config.parse_obj(cfg_values)

    return _config
