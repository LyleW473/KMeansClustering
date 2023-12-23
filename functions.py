import torch

def k_means_cluster(clusters, points):
    complete = False
    iteration_i = 1
    cluster_centers = [cluster.center for cluster in clusters]

    while not complete:
        print(f"Iteration: {iteration_i}")
        print(f"Cluster centers:\n{cluster_centers}\n")

        # Find closest cluster for each point based on their euclidean distance between each cluster center
        point_to_cluster = {} # Maps points to clusters (based on how close they are to the the cluster center)
        for p in points:
            dist_from_cluster_centers = torch.tensor([calc_euclidean_distance(point1 = p, point2 = cluster.center) for cluster in clusters])
            closest_cluster_index = torch.argmin(dist_from_cluster_centers, dim = 0)
            point_to_cluster[p] = closest_cluster_index.item()

        # Reset the points tensors in clusters 
        for cluster in clusters:
            cluster.points = None
        
        # Re-assign points to clusters
        for p in points:
            # Add the point to each cluster
            cluster_index = point_to_cluster[p]
            clusters[cluster_index].add_to_cluster(p)

        # Updating clusters
        complete = True # Assume that the clusters have not changed (i.e., algorithm is "complete" at the beginning)
        for i, cluster in enumerate(clusters):
            # Update cluster center based on the points in the cluster
            cluster.set_cluster_center()
            
            # Check if there are any changes in any of the clusters have changed, setting the flag to exit the algorithm to False if there were any changes
            if torch.equal(cluster.center, cluster_centers[i]) == False:
                complete = False

        # Update list of cluster centers as these will used to compare between cluster centers from this iteration and the next iteration
        cluster_centers = [cluster.center for cluster in clusters]

        if not complete:
            iteration_i += 1
    
    print("------------------------------------------------------------------------------")
    print("Final results:")
    print(f"Total no.of iterations: {iteration_i}")
    print(f"Cluster centers:\n{cluster_centers}\n")

    # Find within-cluster variance of each cluster
    find_within_cluster_variance(clusters = clusters)

def calc_euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1)**2 + (y2 - y1) ** 2)**0.5

def find_within_cluster_variance(clusters):
    # Calculate within-cluster variance of each cluster
    print("Within-cluster variances")
    total_cluster_variance = 0
    for cluster in clusters:
        cluster_variance = cluster.calc_cluster_variance()
        total_cluster_variance += cluster_variance
        print(f"Cluster {cluster.cluster_number}: {cluster_variance}")
    print(f"\nSum of all within-cluster variances of all clusters: {total_cluster_variance}")
