from abc import ABC, abstractmethod
from typing import *

import numpy as np
from .layers import Layer
from .losses import Loss
from .dataset import Dataset


class MultiLayerPerceptron:
    def __init__(self, layers: List[Layer], loss: Loss) -> None:
        self.layers = layers
        self.loss_function = loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        results = np.array([self._forward(sample, train=False) for sample in x])
        return results

    def _forward(self, x: np.ndarray, train: bool) -> np.ndarray:
        intermed_result: np.ndarray = x
        for layer in self.layers:
            intermed_result = layer.forward(intermed_result, train=train)
        return intermed_result

    def _backpropegate(self) -> None:
        gradient = self.loss_function.backwards()
        for layer in self.layers[::-1]:
            gradient = layer.backwards(gradient)

    def _update_weights(self, lr: float = 0.01) -> None:
        for layer in self.layers[::-1]:
            layer.train(lr)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        lr: float = 0.01,
        batch_size: int = 1,
        loss_threshold: float = 1e-2,
        shuffle: bool = True,
    ):
        epoch: int = 1
        while epoch <= epochs:
            losses = []
            xs, ys = x, y
            if shuffle:
                xs, ys = Dataset.shuffle(x, y)
            for sample_x, sample_y in zip(xs, ys):
                result = self._forward(sample_x, train=True)
                loss = self.loss_function.loss(result, sample_y, train=True)
                losses.append(loss)

                self._backpropegate()
                self._update_weights(lr)

            mean_loss = np.mean(np.array(losses))
            if epoch % 100 == 0:
                print(f"mean loss = {mean_loss}")
            if loss_threshold > 0 and mean_loss < loss_threshold:
                break

            epoch += 1

