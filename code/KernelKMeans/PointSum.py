# Author: Jarrid Rector-Brooks

class PointSum:
    def __init__(self, idx, kernelMatrix, clusters):
        self.idx = idx
        self.kernelMatrix = kernelMatrix
        self.clusterSums = []

        for cluster in clusters:
            sum = 0
            for point in cluster.clusterPoints:
                sum += self.kernelMatrix[self.idx, point]

            self.clusterSums.append(sum)

    def addPoint(self, point, clusterNum):
        self.clusterSums[clusterNum] += self.kernelMatrix[self.idx, point]

    def changePointCluster(self, other, oldCluster, newCluster):
        val = self.kernelMatrix[self.idx, other]

        self.clusterSums[oldCluster] -= val
        self.clusterSums[newCluster] += val

    def clusterSum(self, clusterNum):
        return self.clusterSums[clusterNum]
