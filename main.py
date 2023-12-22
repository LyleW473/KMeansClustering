import torch

torch.manual_seed(2004)
K = 3 # Number of clusters
points = [(1, 1), (4, 4), (6, 4), (2, 5), (4, 5), (2, 6), (4, 7)] # Points from video "UL 3 - Example K-Means Clustering"