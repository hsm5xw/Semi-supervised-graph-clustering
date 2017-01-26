from KernelMatrix import KernelMatrix
from Cluster import Cluster
from PointSums import PointSums
from DistanceCalc import DistanceCalc
from pdb import set_trace as st
from sklearn.utils import check_random_state
import numpy as np
import copy

# These methods should be static
class MyKernelKMeans:
    def __init__(self):
        pass

    def __initClusters(self, k, n, oLabels=None):
        if oLabels == None:
            state = check_random_state(None)
            labels = state.randint(k, size=n)
        else:
            labels = oLabels

        clusters = [Cluster(self.gramMatrix) for i in range(0, k)]

        for i in range(0, n):
            clusters[labels[i]].addPoint(i)

        return clusters

    def __initClusterMap(self, clusters):
        clusterMap = {}

        for i in range(0, len(clusters)):
            for assignment in clusters[i].clusterPoints:
                clusterMap[assignment] = i

        return clusterMap

    def __converged(self, clusterMap):
        if clusterMap != self.lastClusterMap:
            self.lastClusterMap = copy.deepcopy(clusterMap)
            return False
        else:
            return True

    def __iterate(self, clusterMap, clusters, pointSums, n, maxIter):
        t = 0
        distCalcAgent = DistanceCalc(self.gramMatrix)

        while (not self.__converged(clusterMap)) and t < maxIter:
            clusterAssignmentChanges = []
            # Determine whether any assignments change
            for i in range(0, n):
                newAssignment = distCalcAgent.minDist(i, clusters, pointSums)
                if newAssignment != clusterMap[i]:
                    clusterAssignmentChanges.append((i, clusterMap[i]))
                    clusterMap[i] = newAssignment

            for dataPointIdx, oldCluster in clusterAssignmentChanges:
                clusters[oldCluster].removePoint(dataPointIdx)
                clusters[clusterMap[dataPointIdx]].addPoint(dataPointIdx)
                pointSums.changeClusterAssignment(dataPointIdx, oldCluster, clusterMap[dataPointIdx])

            t += 1

        print(t)
        clusterAssignments = [clusterMap[i] for i in range(0, n)]
        return clusterAssignments

    def __ensureProperInitialization(self, clusters, pointSums):
        pointsInClusters = {}
        distCalcAgent = DistanceCalc(self.gramMatrix)
        for cluster in clusters:
            for point in cluster.clusterPoints:
                pointsInClusters[point] = True

        assignments = {}
        for i in range(len(pointSums.points)):
            if not i in pointsInClusters:
                assignments[i] = distCalcAgent.minDist(i, clusters, pointSums)

        for pointNum, clusterNum in assignments.items():
            clusters[clusterNum].addPoint(pointNum)
            pointSums.addClusterAssignment(pointNum, clusterNum)


    # Should pointSums use kernel space data or input space data?
    # Change to use alpha
    def cluster(self, gramMatrix, k, maxIter, clusters = None, oLabels = None):
        self.lastClusterMap = None
        self.gramMatrix = gramMatrix
        n = self.gramMatrix.shape[0]
        if clusters == None:
            clusters = self.__initClusters(k, n, oLabels)
            userInitialized = False
        else:
            userInitialized = True

        pointSums = PointSums(self.gramMatrix, clusters, n)
        if userInitialized:
            self.__ensureProperInitialization(clusters, pointSums)

        clusterMap = self.__initClusterMap(clusters)

        return self.__iterate(clusterMap, clusters, pointSums, n, maxIter)
