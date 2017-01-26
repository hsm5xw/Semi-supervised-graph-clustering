import numpy as np

# Author: Jarrid Rector-Brooks

class KernelFunctions:
    @staticmethod
    def Polynomial(x, y, d):
        return np.dot(x, y) ** d

    @staticmethod
    def Gaussian(x, y, sigma):
        return np.exp(-1 * (np.linalg.norm(x - y) ** 2) / (2 * (sigma ** 2)))

    @staticmethod
    def InverseMultiquadratic(x, y, c):
        return 1 / (np.linalg.norm(x - y) ** 2 + c) ** .5

    @staticmethod
    def Linear(x, y):
        return np.dot(x.T, y)
