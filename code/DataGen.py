import numpy as np
import math
import copy
from pdb import set_trace as st

# Author: Jarrid Rector-Brooks

class DataGen:
    @staticmethod
    def GenerateCircle(xCenter, yCenter, radius, size, sigma = .1):
        points = np.zeros((size, 2))
        sample = np.random.normal(0, sigma, size)

        for i in range(0, size):
            # Random angle
            theta = 2 * math.pi * np.random.random()

            r = radius + sample[i]
            x = r * math.cos(theta) + xCenter
            y = r * math.sin(theta) + yCenter

            points[i,0] = x
            points[i,1] = y

        return points

    @staticmethod
    def __findPointClass(point, classRanges):
        for i in range(len(classRanges)):
            if point >= classRanges[i][0] and point < classRanges[i][1]:
                return i

        return -1

    @staticmethod
    def GenerateConstraints(weightMatrix, classRanges, numConstraints, numPoints, k, onlyMustLink, spectral):
        if numConstraints == 0:
            return weightMatrix

        if spectral:
            newMatrix = copy.deepcopy(weightMatrix)
        else:
            newMatrix = weightMatrix

        numbers = []
        labels = {}
        for i in range(len(classRanges)):
            for j in range(classRanges[i][0], classRanges[i][1]):
                numbers.append(j)
                labels[j] = i

        choices = {}
        for i in numbers:
            for j in numbers:
                if i != j:
                    if onlyMustLink:
                        if labels[i] == labels[j]:
                            choices[(i, j)] = True
                    else:
                        choices[(i, j)] = True

        weightVal = numPoints / float(numConstraints * k)

        currNumConstraints = 0
        while currNumConstraints < numConstraints:
            idx = np.random.choice(len(choices.keys()))
            pair = choices.keys()[idx]
            i = pair[0]
            j = pair[1]

            if labels[i] == labels[j]:
                if spectral:
                    newMatrix[i][j] = 1
                    newMatrix[j][i] = 1
                else:
                    newMatrix[i][j] = weightVal
                    newMatrix[j][i] = weightVal

                currNumConstraints += 1
                choices.pop((i,j))
                choices.pop((j,i))

            elif not onlyMustLink:
                if spectral:
                    newMatrix[i][j] = 0
                    newMatrix[j][i] = 0
                else:
                    newMatrix[i][j] = -weightVal
                    newMatrix[j][i] = -weightVal

                currNumConstraints += 1
                choices.pop((i,j))
                choices.pop((j,i))

        return newMatrix

    @staticmethod
    def GenerateTestConstraints():
        constraintMatrix = np.zeros((6,6))
        constraintMatrix[0,1] = 1
        constraintMatrix[1,0] = 1
        constraintMatrix[1,2] = 1
        constraintMatrix[2,1] = 1
        constraintMatrix[3,4] = 1
        constraintMatrix[4,3] = 1
        constraintMatrix[4,5] = 1
        constraintMatrix[5,4] = 1
        constraintMatrix[0,5] = -1

        return constraintMatrix

    @staticmethod
    def TestFFT():
        data = np.zeros((10, 2))
        data[0] = [0,5]
        data[1] = [-1,4]
        data[2] = [1,4]
        data[9] = [0, 4]
        data[3] = [-3,-1]
        data[4] = [-3,-2]
        data[5] = [-1, -8]
        data[6] = [1, -8]
        data[7] = [2, -1]
        data[8] = [2, -2]

        cm = np.zeros((10,10))
        cm[0,1] = 1
        cm[1,2] = 1
        cm[2,9] = 1
        cm[3,4] = 1
        cm[5,6] = 1
        cm[7,8] = 1

        return data, cm

    @staticmethod
    def GenClassRanges(labels):
        lastLabel = -1
        classNum = 0
        ranges = []

        for i in range(len(labels)):
            if labels[i] != lastLabel:
                lastLabel = labels[i]
                ranges.append([i, 0])
                if classNum > 0:
                    ranges[classNum - 1][1] = i - ((i - ranges[classNum - 1][0]) / 2)

                classNum += 1

        ranges[classNum - 1][1] = len(labels) - ((len(labels) - ranges[classNum - 1][0]) / 2)
        return ranges

    @staticmethod
    def GetTestLabels(classRanges, end, labels):
        testLabels = []
        for i in range(1, len(classRanges)):
            testLabels += labels[classRanges[i-1][1]:classRanges[i][0] ]

        return testLabels + labels[classRanges[len(classRanges)-1][1]:]
