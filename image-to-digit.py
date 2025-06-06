import cupy as cp
import idx2numpy
from matplotlib import pyplot as plt
from layers import Dense, TanH
from losses import mse, mse_prime
from network.network import predict, train


network = [
    Dense(2, 5),
    TanH(),
    Dense(5, 5),
    TanH(),
    Dense(5, 1),
    TanH()
]

imagefile = 'train-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

plt.imshow(imagearray[4])


# train(X, Y, network, mse, mse_prime, 10000)

# points = []
# for x in cp.linspace(0, 1, 50):
#     for y in cp.linspace(0, 1, 50):
#         p = predict(network, [[x], [y]])
#         points.append([x, y, p[0][0]])


# points = cp.array(points).get()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2],
#            c=points[:, 2], cmap="winter")
# plt.show()
