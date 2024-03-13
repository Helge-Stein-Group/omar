import numpy as np
from enum import Enum


class Sign(Enum):
    pos = 1
    neg = -1


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
