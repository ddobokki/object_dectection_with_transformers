from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

from literal import DatasetColumns, RawDataColumns


@dataclass
class DataCollatorForObjectDectection:
    feature_extractor: DetrImageProcessor
    padding = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [feature[DatasetColumns.pixel_values] for feature in features]
        # target = [{"image_id": 1, "annotations": annotations}]
        if DatasetColumns.labels in features[0]:
            # labels = [feature[DatasetColumns.labels] for feature in features]
            targets = [
                {
                    "image_id": feature[DatasetColumns.labels][0]["image_id"],
                    "annotations": feature[DatasetColumns.labels],
                }
                for feature in features
            ]
            batch = self.feature_extractor(images=images, annotations=targets, return_tensors="pt")
            # 다음과 같은 형식으로 annotations을 넘겨주어야함
            # targets = {
            #     "image_id": 4538,
            #     "annotations": [
            #         {"area": 119364, "bbox": [155, 89, 348, 343], "category_id": 69, "image_id": 4538, "iscrowd": 0}
            #     ],
            # }
        else:
            batch = self.feature_extractor(images, return_tensors=self.return_tensors)
        return batch
