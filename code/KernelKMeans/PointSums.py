from PointSum import PointSum

# Author: Jarrid Rector-Brooks

class PointSums:
    def __init__(self, kernelMatrix, clusters, n):
        self.points = [PointSum(i, kernelMatrix, clusters) for i in range(0, n)]

    def __getitem__(self, idx):
       return self.points[idx]

    def changeClusterAssignment(self, idx, oldCluster, newCluster):
        for point in self.points:
            point.changePointCluster(idx, oldCluster, newCluster)

    def addClusterAssignment(self, idx, clusterNum):
        for point in self.points:
            point.addPoint(idx, clusterNum)
