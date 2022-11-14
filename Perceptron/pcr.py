import numpy as np
from typing import *
from pprint import pprint
from abc import ABC, abstractmethod


class Perceptron():
	def __init__(self, num_inputs):
		self.weights = np.random.randn(num_inputs + 1, 1)

	def activation(self, input:np.ndarray):
		act = np.zeros((input.shape[0], 1))
		act[input > 0.5] = 1
		return act

	def forward(self, input:np.ndarray):
		input_with_bias = np.hstack([np.ones((input.shape[0],1)), input])
		return self.activation(np.dot(input_with_bias, self.weights))

	def train(self, x:np.ndarray, y:np.ndarray, epochs:int = 1, lr:float = 0.001):
		epoch = 0
		while epoch < epochs:
			y_h = self.forward(x)
			x_b = np.hstack([np.ones((x.shape[0],1)), x])
			delta = lr * (y - y_h) * x_b
			if np.sum(delta) == 0:
				print(epoch)
				break
			for sample_delta in delta:
				self.weights = (self.weights.T + sample_delta).T
			epoch += 1


x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]).reshape(-1,2,1)
y = np.array([
    [0],
    [1],
    [1],
    [0]
])


network:List[Layer] = [
	Linear(2, 2),
	Sigmoid(),
	Linear(2,1),
	Sigmoid()
]
loss = MeanSquereError()

mlp = MultiLayerPerceptron(network, loss)
pred = mlp.predict(x)
print("first try: \n", np.round(pred,0))
mlp.train(x,y, 5000, lr = 1)
pred = mlp.predict(x)
print("second try: \n", np.round(pred,0))




# pcr = Perceptron(2)
# pcr.train(x,y,100, lr=0.33)

# print(pcr.weights)
# print(pcr.forward(np.array([[1,1]])))