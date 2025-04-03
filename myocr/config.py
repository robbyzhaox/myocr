import logging.config
import os

import yaml  # type: ignore

# os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)
MODEL_PATH = (
    os.environ.get("MYOCR_MODEL_PATH")
    or os.path.expanduser("~/.MyOCR/models/")
    or os.path.expanduser("./models")
)

with open(BASE_PATH + "/logging_config.yaml", "r") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
