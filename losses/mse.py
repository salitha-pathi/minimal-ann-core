import cupy as cp


def mse(y, y_pred):
    return cp.mean(cp.power(y - y_pred, 2))


def mse_prime(y, y_pred):
    return 2 * (y_pred - y) / cp.size(y)
