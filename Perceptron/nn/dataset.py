import numpy as np
from typing import *
from abc import ABC, abstractmethod

class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    @classmethod
    def shuffle(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
        sampling = np.random.permutation(x.shape[0])
        return x[sampling], y[sampling]

    def generate_batches(
        self, batch_size: int = 1, shuffle: bool = True
    ) -> List[np.ndarray]:
        num_batches = int(self.x.shape[0] / batch_size)
        data_x, data_y = (self.x, self.y)
        if shuffle:
            data_x, data_y = self.shuffle(self.x, self.y)

        for i in range(num_batches):
            batch_x = data_x[i * batch_size : (i + 1) * batch_size]
            batch_y = data_y[i * batch_size : (i + 1) * batch_size]
            yield (batch_x, batch_y)
