import torch

class Cluster:

    def __init__(self, cluster_number):
        self.cluster_number = cluster_number
        self.points = None
        self.center = None

    # Adds point to cluster
    def add_to_cluster(self, point):

        point_to_add = torch.tensor([point[0], point[1]]).view(1, -1) # Change shape to [1, 2]

        if self.points == None:
            self.points = point_to_add
        else:
            self.points = torch.cat(((self.points, point_to_add)), dim = 0)

    def set_cluster_center(self):
        average_xy = self.points.mean(dim = 0, dtype = torch.float32)
        print(average_xy[0], average_xy[1])
        self.center = torch.tensor([average_xy[0], average_xy[1]])
