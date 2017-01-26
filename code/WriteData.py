import csv
import numpy as np

# Author: Jarrid Rector-Brooks

class DataWrite:
    @staticmethod
    def WriteLabels(labels):
        with open('labels.data', 'w') as labelsFile:
            labelWriter = csv.writer(labelsFile)

            labelWriter.writerow(labels)

    @staticmethod
    def WriteData(data):
        with open('points.data', 'w') as pointsFile:
            pointsWriter = csv.writer(pointsFile)

            pointsWriter.writerow(data.shape)
            for i in range(data.shape[0]):
                pointsWriter.writerow(data[i,:])

    @staticmethod
    def ReadLabels():
        with open('labels.data', 'r') as labelsFile:
            labelReader = csv.reader(labelsFile)

            for row in labelReader:
                return [int(i) for i in row]

    @staticmethod
    def ReadPoints(fname):
        with open(fname, 'r') as pointsFile:
            pointsReader = csv.reader(pointsFile)

            i = 0
            matrix = None
            for row in pointsReader:
                if i == 0:
                    n = int(row[0])
                    m = int(row[1])
                    matrix = np.zeros((n,m))
                else:
                    matrix[i-1,:] = np.array(row, dtype='f8')

                i += 1

            return matrix

