from DataGen import *
from SSKernelKMeans import *
from DataLoad import *
from WriteData import *
from scipy.sparse import csr_matrix as csrm
from scipy.sparse import csgraph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn.metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

# Author: Jarrid Rector-Brooks, Hong Moon

def GetLabelIndices(labels):
    labelMap = {0: [], 1: []}
    for i in range(len(labels)):
        labelMap[labels[i]].append(i)

    return labelMap

def TestWithSynthetic(dataSize, paramVal, constraintStep):
    data = np.zeros((dataSize, 2))
    nmiVals = np.zeros((11, 3))
    trueAssignments = [0 if i < (dataSize / 4) else 1 for i in range(dataSize / 2)]
    numConstraints = np.zeros((11))

    data[0:dataSize / 2, :] = DataGen.GenerateCircle(.5, .5, 2, dataSize / 2)
    data[dataSize / 2:dataSize, :] = DataGen.GenerateCircle(.5, .5, 5, dataSize / 2)

    rbfSimilarityMatrix = sklearn.metrics.pairwise_kernels(data, None, metric='rbf', filter_params=True, gamma=1)
    linearSimilarityMatrix = sklearn.metrics.pairwise_kernels(data, None, metric='linear', filter_params=True, gamma=1)
    classRanges = [(0, dataSize / 4), (dataSize / 2, 3 * dataSize / 4)]

    rbfAverages = []
    linearAverages = []
    wardAverages = []
    # The value inside range indicates the number of iterations to be averaged
    for i in range(2):
        for j in range(11):
            bothLinksConstraintMatrix = np.zeros((dataSize,dataSize))
            onlyMustLinksConstraintMatrix = np.zeros((dataSize,dataSize))
            bothLinksConstraintMatrix = DataGen.GenerateConstraints(bothLinksConstraintMatrix, classRanges, j * constraintStep, dataSize, 2, False, None)
            onlyMustLinksConstraintMatrix = csrm(DataGen.GenerateConstraints(onlyMustLinksConstraintMatrix, classRanges, j * constraintStep, dataSize, 2, True, None))

            wardAgent = AgglomerativeClustering(n_clusters=2, connectivity=onlyMustLinksConstraintMatrix)
            ssKernelKMeansAgent = SSKernelKMeans()

            rbfSSKernelKMeansClusterAssignments = ssKernelKMeansAgent.Cluster(rbfSimilarityMatrix, bothLinksConstraintMatrix, 2)
            linearSSKernelKMeansClusterAssignments = ssKernelKMeansAgent.Cluster(linearSimilarityMatrix, bothLinksConstraintMatrix, 2)
            wardClusterAssignments = wardAgent.fit_predict(data)

            nmiVals[j,0] = sklearn.metrics.normalized_mutual_info_score(trueAssignments, DataGen.GetTestLabels(classRanges, 200, rbfSSKernelKMeansClusterAssignments))
            nmiVals[j,1] = sklearn.metrics.normalized_mutual_info_score(trueAssignments, DataGen.GetTestLabels(classRanges, 200, linearSSKernelKMeansClusterAssignments))
            nmiVals[j,2] = sklearn.metrics.normalized_mutual_info_score(trueAssignments, DataGen.GetTestLabels(classRanges, 200, wardClusterAssignments.tolist()))
            numConstraints[j] = j * constraintStep

            print('SS Kernel K Means with rbf kernel NMI with ' + str(numConstraints[j]) + ' constraints = ' + str(nmiVals[j,0]))
            print('SS Kernel K Means with linear kernel NMI with ' + str(numConstraints[j]) + ' constraints = ' + str(nmiVals[j,1]))
            print('Ward Clustering NMI with ' + str(numConstraints[j]) + ' constraints = ' + str(nmiVals[j,2]))

        rbfAverages.append(nmiVals[:,0])
        linearAverages.append(nmiVals[:,1])
        wardAverages.append(nmiVals[:,2])

    plt.plot(numConstraints, np.mean(rbfAverages, axis=0), '--x')
    plt.plot(numConstraints, np.mean(linearAverages, axis=0), '-o')
    plt.plot(numConstraints, np.mean(wardAverages, axis=0), ':s')

    plt.legend(['SS Kernel KMeans - Gaussian', 'SS Kernel KMeans - Linear', 'Agglomerative Clustering'], loc='upper left')
    plt.xlabel('Number of Constraints')
    plt.ylabel('NMI Value')
    plt.title('Two Circles Data Set')

    plt.show()

def TestLetters():
    features, labels = DataLoad.LoadLetters()
    n = features.shape[0]
    nmiVals = np.zeros((11,3))
    numConstraints = np.zeros((11))

    classRanges = DataGen.GenClassRanges(labels)
    rbfSimilarityMatrix = sklearn.metrics.pairwise_kernels(features, None, metric='rbf', filter_params=True, gamma=.0001)
    linearSimilarityMatrix = sklearn.metrics.pairwise_kernels(features, None, metric='linear', filter_params=True, gamma=1)

    rbfAverages = []
    linearAverages = []
    wardAverages = []
    # The value inside range indicates the number of iterations to be averaged
    for j in range(2):
        for i in range(1,11):
            bothLinksConstraintMatrix = np.zeros((n,n))
            onlyMustLinksConstraintMatrix = np.zeros((n,n))
            bothLinksConstraintMatrix = DataGen.GenerateConstraints(bothLinksConstraintMatrix, classRanges, i * 50, n, 3, False, None)
            onlyMustLinksConstraintMatrix = csrm(DataGen.GenerateConstraints(onlyMustLinksConstraintMatrix, classRanges, i * 50, n, 3, True, None))

            wardAgent = AgglomerativeClustering(n_clusters=3, connectivity=onlyMustLinksConstraintMatrix)
            ssKernelKMeansAgent = SSKernelKMeans()

            rbfSSKernelKMeansClusterAssignments = ssKernelKMeansAgent.Cluster(rbfSimilarityMatrix, bothLinksConstraintMatrix, 3)
            linearSSKernelKMeansClusterAssignments = ssKernelKMeansAgent.Cluster(linearSimilarityMatrix, bothLinksConstraintMatrix, 3)
            wardClusterAssignments = wardAgent.fit_predict(features)

            nmiVals[i,0] = sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, rbfSSKernelKMeansClusterAssignments))
            nmiVals[i,1] = sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, linearSSKernelKMeansClusterAssignments))
            nmiVals[i,2] = sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, wardClusterAssignments.tolist()))
            numConstraints[i] = i * 50
            if nmiVals[i,0] <= 1e-03:
                import pdb
                pdb.set_trace()

            print('SS Kernel K Means with rbf kernel NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,0]))
            print('SS Kernel K Means with linear kernel NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,1]))
            print('Ward Clustering NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,2]))

        rbfAverages.append(nmiVals[:,0])
        linearAverages.append(nmiVals[:,1])
        wardAverages.append(nmiVals[:,2])

    plt.plot(numConstraints, np.mean(rbfAverages, axis=0), '--x')
    plt.plot(numConstraints, np.mean(linearAverages, axis=0), '-o')
    plt.plot(numConstraints, np.mean(wardAverages, axis=0), ':s')

    plt.legend(['SS Kernel KMeans - RBF', 'SS Kernel KMeans - Linear', 'Ward Clustering'], loc='upper left')
    plt.xlabel('Number of Constraints')
    plt.ylabel('NMI Value')
    plt.title('Letters Data Set')

    plt.show()

def TestPendigits():
    features, labels = DataLoad.LoadPendigits()
    n = features.shape[0]
    print(str(n))
    nmiVals = np.zeros((11,3))
    numConstraints = np.zeros((11))

    classRanges = DataGen.GenClassRanges(labels)
    rbfSimilarityMatrix = sklearn.metrics.pairwise_kernels(features, None, metric='rbf', filter_params=True, gamma=.0001)
    linearSimilarityMatrix = sklearn.metrics.pairwise_kernels(features, None, metric='linear', filter_params=True, gamma=1)

    rbfAverages = []
    linearAverages = []
    wardAverages = []
    # The value inside range indicates the number of iterations to be averaged
    for j in range(2):
        for i in range(11):
            bothLinksConstraintMatrix = np.zeros((n,n))
            onlyMustLinksConstraintMatrix = np.zeros((n,n))
            bothLinksConstraintMatrix = DataGen.GenerateConstraints(bothLinksConstraintMatrix, classRanges, i * 50, n, 3, False, None)
            onlyMustLinksConstraintMatrix = csrm(DataGen.GenerateConstraints(onlyMustLinksConstraintMatrix, classRanges, i * 50, n, 3, True, None))

            wardAgent = AgglomerativeClustering(n_clusters=3, connectivity=onlyMustLinksConstraintMatrix)
            ssKernelKMeansAgent = SSKernelKMeans()

            rbfSSKernelKMeansClusterAssignments = ssKernelKMeansAgent.Cluster(rbfSimilarityMatrix, bothLinksConstraintMatrix, 3)
            linearSSKernelKMeansClusterAssignments = ssKernelKMeansAgent.Cluster(linearSimilarityMatrix, bothLinksConstraintMatrix, 3)
            wardClusterAssignments = wardAgent.fit_predict(features)

            nmiVals[i,0] = sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, rbfSSKernelKMeansClusterAssignments))
            nmiVals[i,1] = sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, linearSSKernelKMeansClusterAssignments))
            nmiVals[i,2] = sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, wardClusterAssignments.tolist()))
            numConstraints[i] = i * 50

            print('SS Kernel K Means with rbf kernel NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,0]))
            print('SS Kernel K Means with linear kernel NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,1]))
            print('Ward Clustering NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,2]))

        rbfAverages.append(nmiVals[:,0])
        linearAverages.append(nmiVals[:,1])
        wardAverages.append(nmiVals[:,2])

    plt.plot(numConstraints, np.mean(rbfAverages, axis=0), '--x')
    plt.plot(numConstraints, np.mean(linearAverages, axis=0), '-o')
    plt.plot(numConstraints, np.mean(wardAverages, axis=0), ':s')

    plt.legend(['SS Kernel KMeans - RBF', 'SS Kernel KMeans - Linear', 'Ward Clustering'], loc='upper left')
    plt.xlabel('Number of Constraints')
    plt.ylabel('NMI Value')
    plt.title('Digits Data Set')

    plt.show()

def TestCaltech():
    similarityMatrix, labels = DataLoad.LoadCaltech()
    spectralSimMatrix = sklearn.preprocessing.normalize(similarityMatrix)
    n = similarityMatrix.shape[0]
    nmiVals = np.zeros((21,2))
    numConstraints = np.zeros((21))

    classRanges = DataGen.GenClassRanges(labels)
    normAssocAverages = []
    spectralAverages = []
    # The value inside range indicates the number of iterations to be averaged
    for j in range(1):
        for i in range(21):
            constraintMatrix = np.zeros((n,n))
            spectralConstraintMatrix = np.zeros((n,n))
            constraintMatrix = DataGen.GenerateConstraints(constraintMatrix, classRanges, i * 50, n, 4, False, False)
            spectralConstraintMatrix = DataGen.GenerateConstraints(spectralSimMatrix, classRanges, i * 50, n, 4, False, True)

            ssKernelKMeansAgent = SSKernelKMeans()
            spectralClusteringAgent = SpectralClustering(n_clusters=4, affinity='precomputed')

            spectralAffMatrix = spectralConstraintMatrix - csgraph.laplacian(spectralSimMatrix)
            spectralAffMatrix = (ssKernelKMeansAgent.findSigma(spectralAffMatrix) * np.identity(n)) + spectralAffMatrix

            ssClusterAssignments = ssKernelKMeansAgent.Cluster(similarityMatrix, constraintMatrix, 4)
            spectralClusteringAssignments = spectralClusteringAgent.fit_predict(spectralAffMatrix)

            nmiVals[i,0] = max(0, sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, ssClusterAssignments)))
            nmiVals[i,1] = sklearn.metrics.normalized_mutual_info_score(DataGen.GetTestLabels(classRanges, n, labels.tolist()), DataGen.GetTestLabels(classRanges, n, spectralClusteringAssignments.tolist()))

            numConstraints[i] = i * 50
            print('SS Kernel K Means NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,0]))
            print('Spectral Clustering NMI with ' + str(numConstraints[i]) + ' constraints = ' + str(nmiVals[i,1]))

        normAssocAverages.append(nmiVals[:,0])
        spectralAverages.append(nmiVals[:,1])

    plt.plot(numConstraints, np.mean(normAssocAverages, axis=0), '--x')
    plt.plot(numConstraints, np.mean(spectralAverages, axis=0), ':s')

    plt.legend(['SS Kernel KMeans - Ratio Association', 'Spectral Clustering'], loc='bottom right')
    plt.xlabel('Number of Constraints')
    plt.ylabel('NMI Value')
    plt.title('Caltech Data Set')

    plt.show()

plt.rcParams.update({'font.size': 21})
#TestWithSynthetic(200, .8, 50)
#TestPendigits()
#TestLetters()
TestCaltech()
