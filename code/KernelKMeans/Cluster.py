# Author: Jarrid Rector-Brooks, Hong Moon

class Cluster:
    def __init__(self, kernelMatrix, pointsInCluster = None):
        self.clusterPoints = pointsInCluster if pointsInCluster != None else []
        self.kernelMatrix = kernelMatrix

        self.sum = 0

        for i in self.clusterPoints:
            for j in self.clusterPoints:
                self.sum += kernelMatrix[i,j]

    def addPoint(self, idx):
        for i in self.clusterPoints:
            self.sum += self.kernelMatrix[i, idx]

        self.clusterPoints.append(idx)

        for j in self.clusterPoints:
            self.sum += self.kernelMatrix[idx, j]

    def removePoint(self, idx):
        for i in self.clusterPoints:
            self.sum -= self.kernelMatrix[idx, i]

        self.clusterPoints.remove(idx)

    def thirdTermVal(self):
        sumVal = 0
        for i in self.clusterPoints:
            for j in self.clusterPoints:
                sumVal += self.kernelMatrix[i,j]
        return sumVal / (len(self.clusterPoints) ** 2)
        #return self.sum / (len(self.clusterPoints) ** 2)

    def size(self):
        return len(self.clusterPoints)

    def center(self):
        return self.sum / float(len(self.clusterPoints))
