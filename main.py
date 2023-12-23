import torch
from cluster import Cluster
from functions import k_means_cluster

torch.manual_seed(2004)
K = 3 # Number of clusters
points = [(1, 1), (4, 4), (6, 4), (2, 5), (4, 5), (2, 6), (4, 7), (1, 8)] # Points from video "UL 3 - Example K-Means Clustering"
points = set([torch.tensor(point) for point in points])
clusters = [Cluster(cluster_number = k) for k in range(K)]
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