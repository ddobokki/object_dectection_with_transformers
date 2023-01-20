import os
import re
from ast import literal_eval
from pathlib import Path
from typing import List, Optional
from unicodedata import normalize

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Image
from matplotlib import patches

from literal import DatasetColumns, RawDataColumns


def get_dataset(csv_path: os.PathLike, cast_image=True) -> Dataset:
    df = pd.read_csv(csv_path)
    if RawDataColumns.bbox in df.columns:
        df[RawDataColumns.bbox] = df[RawDataColumns.bbox].apply(literal_eval)

    data_dict = {DatasetColumns.pixel_values: df[RawDataColumns.file_name].tolist()}

    if RawDataColumns.bbox in df.columns and RawDataColumns.category_id in df.columns:
        bbox = df[RawDataColumns.bbox].to_list()
        category_id = df[RawDataColumns.category_id].to_list()
        area = df[RawDataColumns.area].tolist()
        iscrowd = df[RawDataColumns.iscrowd].tolist()
        image_id = df[RawDataColumns.image_id].tolist()

        data_dict[DatasetColumns.labels] = [
            [
                {
                    RawDataColumns.bbox: bbox_,
                    RawDataColumns.category_id: category_id_,
                    RawDataColumns.area: area_,
                    RawDataColumns.iscrowd: iscrowd_,
                    RawDataColumns.image_id: image_id_,
                }
            ]
            for bbox_, category_id_, area_, iscrowd_, image_id_ in zip(bbox, category_id, area, iscrowd, image_id)
        ]
        # data_dict[RawDataColumns.bbox] = boxes
        # data_dict[RawDataColumns.category_id] = category_id
        # data_dict[RawDataColumns.area] = area
        # data_dict[RawDataColumns.iscrowd] = iscrowd
        # data_dict[RawDataColumns.image_id] = image_id

    dataset = Dataset.from_dict(data_dict)
    if cast_image:
        dataset = dataset.cast_column(DatasetColumns.pixel_values, Image())
    return dataset


def transform(raw):
    rtn_data = {}
    rtn_data[DatasetColumns.pixel_values] = raw[DatasetColumns.pixel_values]
    rtn_data[DatasetColumns.labels] = [
        {
            "class_labels": torch.tensor(raw[DatasetColumns.class_labels]),
            "boxes": torch.tensor(raw[DatasetColumns.boxes]).to(torch.float32),
        }
    ]
    return rtn_data


def plot_bboxes(
    image_file: str, bboxes: List[List[float]], xywh: bool = True, labels: Optional[List[str]] = None
) -> None:

    fig = plt.figure()

    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])

    # read and plot the image
    image = plt.imread(image_file)
    plt.imshow(image)

    # Iterate over all the bounding boxes
    for i, bbox in enumerate(bboxes):
        if xywh:
            xmin, ymin, w, h = bbox
        else:
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin

        # add bounding boxes to the image
        box = patches.Rectangle((xmin, ymin), w, h, edgecolor="red", facecolor="none")

        ax.add_patch(box)

        if labels is not None:
            rx, ry = box.get_xy()
            cx = rx + box.get_width() / 2.0
            cy = ry + box.get_height() / 8.0
            l = ax.annotate(
                labels[i], (cx, cy), fontsize=8, fontweight="bold", color="white", ha="center", va="center"
            )
            l.set_bbox(dict(facecolor="red", alpha=0.5, edgecolor="red"))

    plt.axis("off")
