import random
from os import PathLike
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from transformers import (
    PreTrainedTokenizerBase,
    SwinModel,
    TrOCRProcessor,
    Wav2Vec2CTCTokenizer,
)
from transformers.trainer_utils import EvalPrediction

from literal import RawDataColumns


def seed_everything(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
