import numpy as np
from typing import *

from nn.layers import *
from nn.losses import *
from nn.model import MultiLayerPerceptron


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(-1, 2, 1)
y = np.array([[0], [1], [1], [0]])

network: List[Layer] = [Linear(2, 2, "normal"), Sigmoid(), Linear(2, 1, "normal"), Sigmoid()]
loss = MeanSquereError()

mlp = MultiLayerPerceptron(network, loss)
pred = mlp.predict(x)
print("first try: \n", np.round(pred, 0))
mlp.train(x, y, 5000, lr=1, batch_size=2)
pred = mlp.predict(x)
print("second try: \n", np.round(pred, 0))
