from Graph import Graph
import numpy as np
import sys
sys.path.insert(0, '/home/jrectorb/eecs/545/EECS-545-Project/KernelKMeans')
from Cluster import *
from pdb import set_trace as st

# Author: Jarrid Rector-Brooks, Hong Moon

class FarthestFirstInitialization:
    def __init__(self, similarityMatrix):
        self.similarityMatrix = similarityMatrix

    def __distFromSelectedCluster(self, selectedCluster, candidateCluster):
        firstTerm = selectedCluster.thirdTermVal()

        secondTerm = 0
        for i in selectedCluster.clusterPoints:
            for j in candidateCluster.clusterPoints:
                secondTerm += self.similarityMatrix[i][j]
        secondTerm = (-2 * secondTerm) / (selectedCluster.size() * candidateCluster.size())

        thirdTerm = candidateCluster.thirdTermVal()
        distance = firstTerm + secondTerm + thirdTerm

        return distance

    def __distanceMatrix(self, clusters):
        n = len(clusters)
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                D[i,j] =  self.__distFromSelectedCluster(clusters[i], clusters[j])

        return D

    def InitializeClusters(self, neighborhoods, k):
        candidateClusters = []
        neighborhoodNumToClusterNum = {}

        i = 0
        for clusterNum, members in neighborhoods.items():
            if len(members) > 1:
                candidateClusters.append(Cluster(self.similarityMatrix))
                neighborhoodNumToClusterNum[i] = len(candidateClusters) - 1

            i += 1

        bestSize = -1
        largestCluster = -1
        for neighborhoodNum, neighborhoodMembers in neighborhoods.items():
            if neighborhoodNum in neighborhoodNumToClusterNum:
                clusterNum = neighborhoodNumToClusterNum[neighborhoodNum]
                for i in neighborhoodMembers:
                    candidateClusters[clusterNum].addPoint(i)

                if bestSize == -1 or candidateClusters[clusterNum].size() > bestSize:
                    bestSize = candidateClusters[clusterNum].size()
                    largestCluster = clusterNum

        # Make this cleaner
        if k == len(candidateClusters):
            return candidateClusters
        else:

            distanceMatrix = self.__distanceMatrix(candidateClusters)
            selectedClusters = [largestCluster]
            selectedClusterDists = [distanceMatrix[largestCluster]]

            while len(selectedClusters) < k:
                avgDists = np.mean(selectedClusterDists, 0)
                for i in np.argsort(avgDists)[::-1]:
                    if i not in selectedClusters:
                        selectedClusters.append(i)
                        selectedClusterDists.append(distanceMatrix[i])
                        break

        return [candidateClusters[i] for i in selectedClusters]

