from DataGen import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pdb import set_trace as st
from sklearn.metrics.pairwise import pairwise_kernels
import sklearn.metrics
import sys
sys.path.insert(0, '/home/jrectorb/eecs/545/EECS-545-Project/KernelKMeans')
#sys.path.insert(0, '/Users/Emily/School/eecs/545/Project/EECS-545-Project/KernelKMeans')
from KernelKMeans import *
from SSKernelKMeans import *
from DataLoad import *

# Author: Jarrid Rector-Brooks

class Testing:
    @staticmethod
    def TestSSKernelKMeans(labels, similarityMatrix, numConstraints, classRanges, k, numIt):
        n = similarityMatrix.shape[0]

        nmiVals = np.zeros((numIt))
        numConstraints = np.zeros((numIt))

        for i in range(numIt):
            constraintMatrix = DataGen.GenerateConstraintMatrix(classRanges, i * constraintStep, n, k)
            ssKernelKMeansAgent = SSKernelKMeans()
            clusterAssignments = ssKernelKMeansAgent.Cluster(similarityMatrix, constraintMatrix, k)

            nmiVals[i] = sklearn.metrics.normalized_mutual_info_score(labels, clusterAssignments)
            numConstraints[i] = i * constraintStep

        plt.plot(numConstraints, np.mean(nmiVals))
        plt.show()

def TestWithSynthetic(dataSize, paramVal, constraintStep):
    data = np.zeros((dataSize, 2))
    nmiVals = np.zeros((11))
    trueAssignments = [0 if i < (dataSize / 2) else 1 for i in range(dataSize)]
    numConstraints = np.zeros((11))
    data[0:dataSize / 2, :] = DataGen.GenerateCircle(.5, .5, 2, dataSize / 2)
    data[dataSize / 2:dataSize, :] = DataGen.GenerateCircle(.5, .5, 5, dataSize / 2)
    similarityMatrix = KernelMatrix(data, 1, 1)
    for j in range(1, 11):
        #constraintMatrix = DataGen.GenerateConstraintMatrix([(0, dataSize / 2), (dataSize / 2, dataSize)], j * constraintStep, dataSize, 2)
        #ssKernelKMeansAgent = SSKernelKMeans()
        #clusterAssignments = ssKernelKMeansAgent.Cluster(similarityMatrix, constraintMatrix, 2)
        kernelKMeansAgent = MyKernelKMeans()
        theirAssignments = kernelKMeansAgent.cluster(similarityMatrix.raw(), 2, 100)
        #kernelKMeansAssignments = kernelKMeansAgent.cluster(similarityMatrix, 2, 100)

        kernelKMeansVal = sklearn.metrics.normalized_mutual_info_score(trueAssignments, theirAssignments)
        print('Kernel K Means NMI = ' + str(kernelKMeansVal))
        nmiVals[j] = sklearn.metrics.normalized_mutual_info_score(trueAssignments, clusterAssignments)

        numConstraints[j] = j * constraintStep
        print('NMI with ' + str(numConstraints[j]) + ' constraints = ' + str(nmiVals[j]))

        #plt.scatter(data[:,0], data[:,1], c=clusterAssignments, cmap=cm.Set1)
        #plt.show(block=False)

    plt.plot(numConstraints, nmiVals)
    print('NMI = ' + str(np.average(nmiVals)))
    plt.show()

def TestLetters():
    features, labels = DataLoad.LoadLetters()
    n = features.shape[0]
    nmiVals = np.zeros((11))
    numConstraints = np.zeros((11))

    classRanges = DataGen.GenClassRanges(labels)
    similarityMatrix = KernelMatrix(features, 1, .3)

    averages = []
    for j in range(2):
        for i in range(1):
            constraintMatrix = DataGen.GenerateConstraintMatrix(classRanges, i * 50, n, 3)

            kernelKMeansAgent = KernelKMeans()
            ssKernelKMeansAgent = SSKernelKMeans()
            ssClusterAssignments = ssKernelKMeansAgent.Cluster(similarityMatrix, constraintMatrix, 3)
            kernelKMeansAssignments = kernelKMeansAgent.cluster(similarityMatrix, 3, 100)

            nmiVals[i] = sklearn.metrics.normalized_mutual_info_score(labels, ssClusterAssignments)
            kernelKMeansVal = sklearn.metrics.normalized_mutual_info_score(labels, kernelKMeansAssignments)
            numConstraints[i] = i * 50
            print('NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i]))
            print('Kernel K Means NMI = ' + str(kernelKMeansVal))

        averages.append(nmiVals)

    plt.plot(numConstraints, np.mean(averages))
    print('NMI = ' + str(np.average(nmiVals)))
    plt.show()

def TestPendigits():
    features, labels = DataLoad.LoadPendigits()
    n = features.shape[0]
    nmiVals = np.zeros((11))
    numConstraints = np.zeros((11))

    classRanges = DataGen.GenClassRanges(labels)
    similarityMatrix = pairwise_kernels(features, None, metric='rbf',
                                        filter_params=True, gamma=.00001)

    averages = []
    for j in range(10):
        nmiVals = []
        for i in range(11):
            constraintMatrix = DataGen.GenerateConstraintMatrix(classRanges, i * 50, n, 3)

            ssKernelKMeansAgent = SSKernelKMeans()
            ssClusterAssignments = ssKernelKMeansAgent.Cluster(similarityMatrix, constraintMatrix, 3)

            nmiVals.append(sklearn.metrics.normalized_mutual_info_score(labels, ssClusterAssignments))
            numConstraints[i] = i * 50
            print('NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i]))

        averages.append(nmiVals)

    plt.plot(numConstraints, np.mean(averages, axis=0))
    print('NMI = ' + str(np.average(nmiVals)))
    plt.show()

def TestCaltech():
    similarityMatrix, labels = DataLoad.LoadCaltech()
    n = similarityMatrix.shape[0]
    nmiVals = np.zeros((11))
    numConstraints = np.zeros((11))

    classRanges = DataGen.GenClassRanges(labels)
    averages = []
    for j in range(2):
        nmiVals = [0 for i in range(11)]
        for i in range(1, 11):
            constraintMatrix = DataGen.GenerateConstraintMatrix(classRanges, 500 + (i * 50), n, 3)

            ssKernelKMeansAgent = SSKernelKMeans()
            ssClusterAssignments = ssKernelKMeansAgent.Cluster(similarityMatrix, constraintMatrix, 3)

            nmiVals[i] = sklearn.metrics.normalized_mutual_info_score(labels, ssClusterAssignments)
            numConstraints[i] = 500 + (i * 50)
            print('NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i]))

        averages.append(nmiVals)

    plt.plot(numConstraints, np.mean(averages, axis=0))
    print('NMI = ' + str(np.average(nmiVals)))
    plt.show()

#TestWithSynthetic(200, .8, 50)
#TestPendigits()
#TestLetters()
TestCaltech()
