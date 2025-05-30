import cupy as cp
from .activation import Activation


class TanH(Activation):
    def __init__(self):
        def tanh(x): return cp.tanh(x)
        def tanh_prime(x): return 1 - cp.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)
