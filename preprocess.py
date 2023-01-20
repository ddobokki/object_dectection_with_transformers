import json
import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from literal import Folder, RawDataColumns

with open("data/food_train_anno.json", "r") as json_file:
    train_anno = json.load(json_file)

image_df = pd.DataFrame(train_anno["images"]).set_index(RawDataColumns.id)
annotations_df = pd.DataFrame(train_anno["annotations"]).set_index(RawDataColumns.id)


train_df = pd.merge(image_df, annotations_df, left_index=True, right_index=True).reset_index(drop=True)

train_df[RawDataColumns.file_name] = train_df[RawDataColumns.file_name].apply(
    lambda x: os.path.join(Folder.data_train, x)
)


def rerange_bbox(bbox: List[int]) -> List[int]:
    x1, y1, w, h = bbox
    return [x1, y1, x1 + w, y1 + h]


# train_df[RawDataColumns.bbox] = train_df[RawDataColumns.bbox].apply(rerange_bbox)
if not os.path.exists(Folder.data_preprocess):
    os.mkdir(Folder.data_preprocess)


train_split_df, valid_split_df = train_test_split(
    train_df, test_size=0.05, random_state=42, stratify=train_df[RawDataColumns.category_id]
)

train_split_df = train_split_df.reset_index(drop=True)
valid_split_df = valid_split_df.reset_index(drop=True)
train_split_df.to_csv(os.path.join(Folder.data_preprocess, "train.csv"), index=False)
valid_split_df.to_csv(os.path.join(Folder.data_preprocess, "valid.csv"), index=False)
