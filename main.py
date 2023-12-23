import torch
from cluster import Cluster
from functions import k_means_cluster, find_within_cluster_variance

torch.manual_seed(2004)
K = 3 # Number of clusters
points = [(1, 1), (4, 4), (6, 4), (2, 5), (4, 5), (2, 6), (4, 7), (1, 8)] # Points from video "UL 3 - Example K-Means Clustering"
points = set([torch.tensor(point) for point in points])
clusters = [Cluster(cluster_number = k + 1) for k in range(K)]
random_initialisation = False

# Assigning based on the video
if not random_initialisation:
    clusters[0].center = torch.tensor([2, 6])
    clusters[1].center = torch.tensor([2, 5])
    clusters[2].center = torch.tensor([6, 4])
    print(clusters[0].center)
    print(clusters[1].center)
    print(clusters[2].center)

# Random assignment of points to clusters
else:
    # Randomly assign points to cluster
    for point in points:
        random_cluster_index = torch.randint(low = 0, high = K, size = (1,))
        clusters[random_cluster_index].add_to_cluster(point)
    
    # For each cluster, set the cluster center as the average
    for i, cluster in enumerate(clusters):
        print(f"Cluster number: {cluster.cluster_number}")
        cluster.set_cluster_center()
        print(cluster.points)
        print(cluster.center, "\n")

# Perform K-Means clustering algorithm
k_means_cluster(clusters = clusters, points = points.copy())

# Within cluster variance for UL 1:
# From the Quiz UL 1:
print("------------------------------------------------------------------------------")
print(f"UL 1 - Motivation for K-Means Clustering:\n")

points2 = [(1, 2), (2, 3), (3, 0), (4, 4), (6, 7), (7, 6)]
points2 = [torch.tensor(point) for point in points2]

cluster_C1_tilde = Cluster(cluster_number = 1)
cluster_C1_tilde.add_to_cluster(points2[0])
cluster_C1_tilde.add_to_cluster(points2[1])
cluster_C1_tilde.add_to_cluster(points2[2])
cluster_C1_tilde.add_to_cluster(points2[3])

cluster_C2_tilde = Cluster(cluster_number = 2)
cluster_C2_tilde.add_to_cluster(points2[4])
cluster_C2_tilde.add_to_cluster(points2[5])

find_within_cluster_variance(clusters = [cluster_C1_tilde, cluster_C2_tilde])

cluster_C1_hat = Cluster(cluster_number = 1)
cluster_C1_hat.add_to_cluster(points2[0])
cluster_C1_hat.add_to_cluster(points2[1])
cluster_C1_hat.add_to_cluster(points2[2])

cluster_C2_hat = Cluster(cluster_number = 2)
cluster_C2_hat.add_to_cluster(points2[3])
cluster_C2_hat.add_to_cluster(points2[4])
cluster_C2_hat.add_to_cluster(points2[5])

find_within_cluster_variance(clusters = [cluster_C1_hat, cluster_C2_hat])