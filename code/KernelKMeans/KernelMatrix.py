from KernelFunctions import KernelFunctions
import numpy as np

#  Ideas
#
# Create buckets for sums in 2nd and 3rd terms of kernel k means dist calculation
# When a node shifts clusters, move its K_ij val from its old cluster to its new cluster
# This saves n^2 - cn computations per iteration of k-means where c is the number of nodes
# which change clusters in a given iteration and n is the number of points
#

class KernelMatrix:
    def __init__(self, data, kernelType, paramVal = None):
        self.matrix = None

        if kernelType == 0:
            self.__initMatrix(data, KernelFunctions.Polynomial, paramVal)
        elif kernelType == 1:
            self.__initMatrix(data, KernelFunctions.Gaussian, paramVal)
        elif kernelType == 2:
            self.__initMatrix(data, KernelFunctions.InverseMultiquadratic, paramVal)
        elif kernelType == 3:
            self.__initMatrix(data, KernelFunctions.Linear)

    def __initMatrix(self, data, kernelFunction, paramVal):
        n = data.shape[0]
        self.matrix = np.zeros((n, n))

        for i in range(0, n):
            for j in range(0, n):
                self.matrix[i, j] = kernelFunction(data[i,:], data[j,:], paramVal)

    def __getitem__(self, idx):
        return self.matrix[idx]

    def raw(self):
        return self.matrix



