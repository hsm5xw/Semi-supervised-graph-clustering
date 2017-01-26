from KernelMatrix import KernelMatrix
from pdb import set_trace
import numpy as np

# Author: Jarrid Rector-Brooks, Hong Moon

class DistanceCalc:
    def __init__(self, kernelMatrix):
        self.kernelMatrix = kernelMatrix

    def minDist(self, idx, clusters, pointSums):
        minDist = -1
        bestCluster = -1

        for i in range(0, len(clusters)):
            firstTerm = self.kernelMatrix[idx, idx]
            #if clusters[i].size() == 0:
                #set_trace()
            secondTerm = 2 * pointSums[idx].clusterSum(i) / clusters[i].size()
            thirdTerm = clusters[i].thirdTermVal()
            thisResult = firstTerm - secondTerm + thirdTerm

            if thisResult < minDist or minDist == -1:
                minDist = thisResult
                bestCluster = i

        return bestCluster


