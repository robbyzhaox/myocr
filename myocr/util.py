import hashlib
import os
from urllib.request import urlretrieve
from zipfile import ZipFile

from pathlib2 import Path

from .config import MODULE_PATH


def get_model_path(model_config):
    model_storage_directory = MODULE_PATH + "/model"
    return os.path.join(model_storage_directory, model_config["filename"])


def load_model(model_config):
    model_storage_directory = MODULE_PATH + "/model"
    Path(model_storage_directory).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_storage_directory, model_config["filename"])
    
    if os.path.exists("/home/robby/.MyOCR//model/pretrained_ic15_res18.pt"):
        return
    
    download_and_unzip(
        model_config["url"],
        model_config["filename"],
        model_storage_directory,
    )
    assert calculate_md5(model_path) == model_config["md5sum"]


def download_and_unzip(url, filename, model_storage_directory, verbose=True):
    zip_path = os.path.join(model_storage_directory, "temp.zip")
    reporthook = (
        printProgressBar(prefix="Progress:", suffix="Complete", length=50) if verbose else None
    )
    urlretrieve(url, zip_path, reporthook=reporthook)
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extract(filename, model_storage_directory)
    os.remove(zip_path)


def calculate_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def printProgressBar(prefix="", suffix="", decimals=1, length=100, fill="█"):
    """
    Call in a loop to create terminal progress bar
    """

    def progress_hook(count, blockSize, totalSize):
        progress = count * blockSize / totalSize
        percent = ("{0:." + str(decimals) + "f}").format(progress * 100)
        filledLength = int(length * progress)
        bar = fill * filledLength + "-" * (length - filledLength)
        print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="")

    return progress_hook
