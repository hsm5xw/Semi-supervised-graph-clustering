import numpy as np
import scipy.io as sio
import pdb
import csv

# Author: Jarrid Rector-Brooks

class DataLoad:
    @staticmethod
    def LoadPendigits():
        dataSetDict = {3: [], 8: [], 9: []}
        dataSetList = []
        labels = []
        with open('Digits/pendigits.tes', 'rb') as pendigits:
            loadedDataSet = csv.reader(pendigits)
            for row in loadedDataSet:
                classNum = int(row[16])

                if (classNum == 3 or classNum == 8 or classNum == 9) and np.random.ranf() <= .1:
                    dataSetDict[classNum].append(np.array([int(row[i]) for i in range(16)], dtype=object))

        for classNum, classFeatures in dataSetDict.items():
            for features in classFeatures:
                dataSetList.append(features)
                labels.append(classNum)

        return np.vstack(dataSetList), np.array(labels)

    @staticmethod
    def LoadLetters():
        dataSetDict = {'I': [], 'J': [], 'L': []}
        dataSetList = []
        labels = []
        with open('Letters/letter-recognition.data', 'rb') as letters:
            loadedDataSet = csv.reader(letters)
            for row in loadedDataSet:
                classLabel = row[0]

                if (classLabel == 'I' or classLabel == 'J' or classLabel == 'L') and np.random.ranf() <= .1:
                    dataSetDict[classLabel].append(np.array([int(row[i]) for i in range(1, 17)], dtype=object))

        for classNum, classFeatures in dataSetDict.items():
            for features in classFeatures:
                dataSetList.append(features)
                labels.append(classNum)

        return np.vstack(dataSetList), np.array(labels)

    @staticmethod
    def LoadCaltech():
        labels = []
        for i in range(300):
            if i < 75:
                labels.append(0)
            elif i < 150:
                labels.append(1)
            elif i < 225:
                labels.append(2)
            else:
                labels.append(3)

        return sio.loadmat('kMatrix.mat')['K'], np.array(labels)

