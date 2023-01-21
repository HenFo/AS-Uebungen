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
    def __init__(self, num_inputs: int, num_outputs: int, weight_init:str = "uniform") -> None:
        assert weight_init in ("uniform", "normal")

        self.num_inputs = num_inputs
        self.num_inputs = num_outputs

        if weight_init == "uniform":
            self.weights = np.random.rand(num_outputs, num_inputs + 1)
        if weight_init == "normal":
            self.weights = np.random.randn(num_outputs, num_inputs + 1)
        
        # self.weights *= np.sqrt(1.0 / num_inputs)

        self.weight_gradients: np.ndarray = None

        self.train_input: np.ndarray = None
        self.train_output: np.ndarray = None

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        x_bias = np.vstack([np.ones((1, 1)), x])
        out = np.dot(self.weights, x_bias)
        if train:
            self.train_input = x_bias
            self.train_output = out
        return out

    def backwards(self, prev_grads: np.ndarray) -> np.ndarray:
        self.weight_gradients = np.dot(prev_grads, self.train_input.T)
        return np.dot(self.weights[:,1:].T, prev_grads)

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

class LeakyReLu(Layer):
    def __init__(self, leak:float = 0.2) -> None:
        self.train_input: np.ndarray = None
        self.train_output: np.ndarray = None
        self.leak:float = leak
    
    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        out = x.copy()
        out[out < 0] *= self.leak
        if train:
            self.train_input = x
            self.train_output = out
        return out

    def backwards(self, prev_grads: np.ndarray) -> np.ndarray:
        grads = np.ones_like(self.train_input)
        grads[self.train_output == 0] = self.leak
        return prev_grads * grads

    def train(self, *_) -> None:
        self.train_input = None
        self.train_output = None


class Tanh(Layer):
    def __init__(self) -> None:
        self.train_input: np.ndarray = None
        self.train_output: np.ndarray = None
    
    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        out = np.tanh(x)
        if train:
            self.train_input = x
            self.train_output = out
        return out

    def backwards(self, prev_grads: np.ndarray) -> np.ndarray:
        grads = 2 / (np.cosh(2 * self.train_input) + 1)
        return prev_grads * grads

    def train(self, *_) -> None:
        self.train_input = None
        self.train_output = None