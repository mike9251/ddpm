import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")