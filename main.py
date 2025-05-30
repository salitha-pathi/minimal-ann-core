from matplotlib import pyplot as plt
from dense import Dense
from losses import mse, mse_prime
from network import predict, train
from tanh import TanH
import cupy as np


network = [
    Dense(2, 3),
    TanH(),
    Dense(3, 1),
    TanH()
]

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], newshape=(4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], newshape=(4, 1, 1))


train(X, Y, network, mse, mse_prime)

points = []
for x in np.linspace(0, 1, 50):
    for y in np.linspace(0, 1, 50):
        p = predict(network, [[x], [y]])
        points.append([x, y, p[0][0]])


points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           c=points[:, 2], cmap="winter")
plt.show()
