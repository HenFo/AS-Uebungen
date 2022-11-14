import numpy as np
from typing import *
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def backwards(self, prev_grads: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def train(self, lr: float = 0.01) -> None:
        raise NotImplementedError()


class Linear(Layer):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        self.num_inputs = num_inputs
        self.num_inputs = num_outputs

        self.weights = np.random.randn(num_inputs + 1, num_outputs)
        self.weight_gradients: np.ndarray = None

        self.train_input: np.ndarray = None
        self.train_output: np.ndarray = None

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        x_bias = np.vstack([np.ones((1, 1)), x])
        out = np.dot(self.weights.T, x_bias)
        if train:
            self.train_input = x_bias
            self.train_output = out
        return out

    def backwards(self, prev_grads: np.ndarray) -> np.ndarray:
        self.weight_gradients = np.dot(prev_grads, self.train_input.T).T
        return np.dot(self.weights[1:], prev_grads)

    def train(self, lr: float = 0.01) -> None:
        self.weights -= lr * self.weight_gradients
        self.weight_gradients = None
        self.train_input = None
        self.train_output = None


class Sigmoid(Layer):
    def __init__(self) -> None:
        self.train_input: np.ndarray = None
        self.train_output: np.ndarray = None

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        if train:
            self.train_input = x
            self.train_output = out
        return out

    def backwards(self, prev_grads: np.ndarray) -> np.ndarray:
        grads = self.train_output * (1 - self.train_output)
        return prev_grads * grads

    def train(self, *_) -> None:
        self.train_input = None
        self.train_output = None
