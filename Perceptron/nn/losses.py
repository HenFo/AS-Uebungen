import numpy as np
from typing import *
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def loss(self, x: np.ndarray, y: np.ndarray, train: bool = False) -> float:
        raise NotImplementedError()

    @abstractmethod
    def backwards(self) -> np.ndarray:
        raise NotImplementedError()


class MeanSquereError(Loss):
    def __init__(self) -> None:
        self.train_x: np.ndarray = 0
        self.train_y: np.ndarray = 0

    def loss(self, x: np.ndarray, y: np.ndarray, train: bool = False) -> float:
        if train:
            self.train_x = x
            self.train_y = y
        return np.mean(0.5 * np.power(x - y, 2))

    def backwards(self) -> np.ndarray:
        out = self.train_x - self.train_y
        return out