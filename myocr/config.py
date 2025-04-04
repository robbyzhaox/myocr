import logging.config
import os

import yaml  # type: ignore

# os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)
WORK_DIR = os.path.dirname(BASE_PATH)


def get_model_path() -> str:
    if "MYOCR_MODEL_PATH" in os.environ and os.environ["MYOCR_MODEL_PATH"]:
        return os.environ["MYOCR_MODEL_PATH"]

    user_path = os.path.expanduser("~/.MyOCR/models/")
    if os.path.exists(user_path):
        return str(user_path)

    return str(os.path.join(WORK_DIR, "models/"))


MODEL_PATH = get_model_path()

with open(BASE_PATH + "/logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
