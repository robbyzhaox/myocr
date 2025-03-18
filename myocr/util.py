import hashlib
import os
from typing import Any
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from pathlib2 import Path

from .config import MODULE_PATH


def get_model_path(model_config):
    model_storage_directory = MODULE_PATH + "/model"
    return os.path.join(model_storage_directory, model_config["filename"])


def load_model(model_config):
    model_storage_directory = MODULE_PATH + "/model"
    Path(model_storage_directory).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_storage_directory, model_config["filename"])

    if os.path.exists("/home/robby/.MyOCR//model/pretrained_ic15_res18.pt") and os.path.exists(
        "/home/robby/.MyOCR//model/zh_sim_g2.pth"
    ):
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


def diff(input_list):
    return max(input_list) - min(input_list)


def group_text_box(
    polys,
    slope_ths=0.1,
    ycenter_ths=0.5,
    height_ths=0.5,
    width_ths=1.0,
    add_margin=0.05,
    sort_output=True,
):
    # poly top-left, top-right, low-right, low-left
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append(
                [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min]
            )
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])

            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
            theta24 = abs(np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box: list[Any] = []
    for poly in horizontal_list:
        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if (abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)) and (
                        (box[0] - x_max) < width_ths * (box[3] - box[2])
                    ):  # merge boxes
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        b_height = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0:
                merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append(
                        [x_min - margin, x_max + margin, y_min - margin, y_max + margin]
                    )
                else:  # non adjacent box in same line
                    box = mbox[0]

                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append(
                        [box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin]
                    )
    # may need to check if box is really in image
    return merged_list, free_list
