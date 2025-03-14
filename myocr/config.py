import os

# os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = (
    os.environ.get("MYOCR_MODULE_PATH")
    or os.environ.get("MODULE_PATH")
    or os.path.expanduser("~/.MyOCR/")
)
