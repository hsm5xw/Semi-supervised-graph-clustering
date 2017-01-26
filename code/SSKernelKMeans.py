from FarthestFirstInitialization import *
from scipy.sparse import csgraph
import numpy as np
import sys
sys.path.insert(0, 'KernelKMeans/')
from KernelKMeans import *
from pdb import set_trace as st

# Author: Jarrid Rector-Brooks, Hong Moon

class SSKernelKMeans:
    def __init__(self):
        pass

    def findSigma(self, k):
        minEigVal = np.amin(np.linalg.eigvals(k))
        if minEigVal < 0:
            return abs(minEigVal)
        else:
            return 0

    def __getMustLinkMatrix(self, weightMatrix):
        n = weightMatrix.shape[0]
        newMatrix = np.zeros(weightMatrix.shape)

        for i in range(n):
            for j in range(n):
                if weightMatrix[i][j] > 0:
                    newMatrix[i][j] = 1

        return newMatrix

    def __neighborhoodsHaveCannotLink(self, a, b, constraintMatrix):
        for i in a:
            for j in b:
                if constraintMatrix[i, j] < 0:
                    return True

        return False


    def __formNeighborhoods(self, closure):
        numComponents, componentLabels = csgraph.connected_components(csgraph.csgraph_from_dense(closure), directed=False)

        neighborhoods = {}
        newConstraints = 0
        for i in range(closure.shape[0]):
            if componentLabels[i] in neighborhoods:
                neighborhoods[componentLabels[i]].append(i)
            else:
                neighborhoods[componentLabels[i]] = [i]

        return neighborhoods

    # For both augmentations, if optimizations are needed, many computations are unnecessary
    def __augmentMustLinkConstraints(self, constraintMatrix, neighborhoods):
        for neighborhoodNum, neighborhoodMembers in neighborhoods.items():
            for i in neighborhoodMembers:
                for j in neighborhoodMembers:
                    if constraintMatrix[i][j] == 0 and i != j:
                        constraintMatrix[i][j] = 1
                        constraintMatrix[j][i] = 1

        return constraintMatrix

    def __augmentCannotLinkConstraints(self, constraintMatrix, neighborhoods):
        for neighborhoodNumI, membersI in neighborhoods.items():
            for neighborhoodNumJ, membersJ in neighborhoods.items():
                if neighborhoodNumI != neighborhoodNumJ and self.__neighborhoodsHaveCannotLink(membersI, membersJ, constraintMatrix):
                    for i in membersI:
                        for j in membersJ:
                            constraintMatrix[i][j] = -1
                            constraintMatrix[j][i] = -1

        return constraintMatrix

    def __reweightConstraintMatrix(self, constraintMatrix, k):
        numConstraints = 0

        for i in range(constraintMatrix.shape[0]):
            for j in range(constraintMatrix.shape[0]):
                if  constraintMatrix[i,j] != 0:
                    numConstraints += 1
        numConstraints /= 2

        # Ensure float division
        weightVal = constraintMatrix.shape[0] / float(k * numConstraints)

        for i in range(constraintMatrix.shape[0]):
            for j in range(constraintMatrix.shape[0]):
                if constraintMatrix[i][j] > 0:
                    constraintMatrix[i][j] = weightVal
                elif constraintMatrix[i][j] < 0:
                    constraintMatrix[i][j] = -weightVal

        return constraintMatrix

    def __hasConstraints(self, constraintMatrix):
        n, d = constraintMatrix.shape

        for i in range(n):
            for j in range(d):
                if constraintMatrix[i][j] != 0:
                    return True

        return False

    def __augmentConstraintMatrix(self, constraintMatrix, neighborhoods, k):
        if self.__hasConstraints(constraintMatrix):
            constraintMatrix = self.__augmentMustLinkConstraints(constraintMatrix, neighborhoods)
            constraintMatrix = self.__augmentCannotLinkConstraints(constraintMatrix, neighborhoods)
            constraintMatrix = self.__reweightConstraintMatrix(constraintMatrix, k)

        return constraintMatrix

    def Cluster(self, similarityMatrix, constraintMatrix, k, maxIt = 100):
        if self.__hasConstraints(constraintMatrix):
            transClosure = Graph.TransitiveClosure(self.__getMustLinkMatrix(constraintMatrix))
            neighborhoods = self.__formNeighborhoods(transClosure)
            constraintMatrix = self.__augmentConstraintMatrix(constraintMatrix, neighborhoods, k)

        initKMatrix = similarityMatrix + constraintMatrix
        # If things are faulty, check here. Min eigval != 0 after diagonal shift right now.
        kMatrix = (self.findSigma(initKMatrix) * np.identity(initKMatrix.shape[0])) + initKMatrix

        if self.__hasConstraints(constraintMatrix):
            initializationAgent = FarthestFirstInitialization(kMatrix)
            initialClusters = initializationAgent.InitializeClusters(neighborhoods, k)
        else:
            initialClusters = None

        # Change!
        kernelKMeansAgent = MyKernelKMeans()

        # Should eventually be this once KernelKMeans is changed to use alpha
        # return kernelKMeansAgent.cluster(kMatrix, k, np.ones(kMatrix.shape[0]), initialClusters)
        return kernelKMeansAgent.cluster(kMatrix, k, maxIt, initialClusters)

