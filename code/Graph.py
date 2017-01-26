from pdb import set_trace as st
import scipy.sparse.csgraph as csgraph
import numpy as np

# Author: Jarrid Rector-Brooks, Hong Moon

class Graph:
    @staticmethod
    def TransitiveClosure(adjMatrix):
        n = len(adjMatrix)

        newMatrix = np.zeros(adjMatrix.shape)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        newMatrix[i][j] = newMatrix[i][j] != 0 or adjMatrix[i][j] != 0 or ((adjMatrix[i][k] == adjMatrix[k][j]) and adjMatrix[i][k] != 0)

        return newMatrix

