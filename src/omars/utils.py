from enum import Enum

import numpy as np


class Sign(Enum):
    pos = 1
    neg = -1


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def hinge(x: np.ndarray):
    return np.maximum(x, np.zeros(x.shape))
