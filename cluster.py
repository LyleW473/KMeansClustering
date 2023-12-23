import torch
from functions import calc_euclidean_distance

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
        average_xy = self.points.mean(dim = 0, dtype = torch.float64)
        self.center = torch.tensor([average_xy[0], average_xy[1]])

    def calc_cluster_variance(self):
        num_points = len(self.points)
        total_distance_of_all_points = 0 # Could have used "sum_distance_with_other_points", but used this extra variable for clarity.

        for i in range(0, num_points):
            current_point = self.points[i]
            sum_distance_with_other_points = 0

            # Find distance with this point and all other points
            for j in range(i + 1, num_points):
                other_point = self.points[j]
                distance_between_points = (calc_euclidean_distance(point1 = current_point, point2 = other_point) ** 2)
                sum_distance_with_other_points += distance_between_points

            # Add to total distance of all points with all other points in the cluster
            total_distance_of_all_points += sum_distance_with_other_points
        
        # Return (1/|Ck|) * Sum of distances between points in the cluster
        return (total_distance_of_all_points / num_points)