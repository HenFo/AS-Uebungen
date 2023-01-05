import numpy as np
import matplotlib.pyplot as plt

NUM_POINTS = 100
NUM_CLASSES = 3

codebooks = np.random.normal(0,1, size=(NUM_CLASSES,2))
points_uni = np.random.uniform(0.5, 1.5, size=(NUM_POINTS, 2))
points_norm = np.random.normal([-1,1], [0.5,0.25], size=(NUM_POINTS,2))

r = 0.5 * np.sqrt(np.random.rand(NUM_POINTS,1))
theta = np.random.rand(NUM_POINTS,1) * 2 * np.pi
x = 0.5 + r * np.cos(theta)
y = -0.5 + r * np.sin(theta)

points_circle = np.hstack([x,y])

all_points = np.vstack([points_circle, points_norm, points_uni])


# # plt.scatter(points_uni[:,0], points_uni[:,1])
# # plt.scatter(points_norm[:,0], points_norm[:,1])
# # plt.scatter(points_circle[:,0], points_circle[:,1])
# # plt.scatter(codebooks[:,0], codebooks[:,1], c="black")
# # plt.grid(True)
# # plt.show()

def get_closest_codebook(points:np.ndarray, codebooks:np.ndarray) -> np.ndarray:
	distances = np.linalg.norm(np.array([[point - codebook for codebook in codebooks] for point in points]), axis=-1)
	closest_codebooks = np.argmin(distances, axis=1)
	return closest_codebooks

classification = get_closest_codebook(all_points, codebooks)

def update_codebook(points:np.ndarray, classifications:np.ndarray):
	means = []
	for i in range(NUM_CLASSES):
		class_points = points[classifications == i]
		mean = np.mean(class_points, axis=0)
		means.append(mean)
	return np.array(means)

codebooks = update_codebook(all_points, classification)

