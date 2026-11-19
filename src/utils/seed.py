"""
Author     : Cao Junjia

Reproducibility utilities.
Call fix_seed(42) at the very beginning of any train/test script.
"""
import os
import random
import numpy as np
import torch


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Makes CUDA ops deterministic (slight speed penalty)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
