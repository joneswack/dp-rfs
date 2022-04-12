import torch
import tqdm

from models.het_gp import HeteroskedasticGP

def cluster_points(data, num_clusters=10, method='random', global_max_dist=1.):

    shuffled_data = data[torch.randperm(len(data))]

    # determine cluster centers
    if method=='farthest':
        cluster_centers = [shuffled_data.mean(dim=0)]

        for _ in range(num_clusters-1):
            distances = torch.cdist(shuffled_data, torch.stack(cluster_centers, dim=0), p=2)

            # distances to the closest cluster centers
            min_dists = distances.min(dim=1)[0]

            if min_dists.max() <= global_max_dist:
                break

            farthest_point = min_dists.argmax()
            cluster_centers.append(shuffled_data[farthest_point])

        print('Number of clusters found: {}'.format(len(cluster_centers)))
        cluster_centers = torch.stack(cluster_centers, dim=0)
    elif method == 'random':
        cluster_centers = shuffled_data[:num_clusters]

    return cluster_centers


def compute_local_predictions(
        train_data, test_data, train_labels,
        cluster_assignments, cluster_centers,
        feature_encoder, noise_var):

    predictive_means = torch.zeros(
        len(test_data), 1,
        dtype=train_data.dtype, device=train_data.device
    )
    predictive_stds = torch.zeros_like(predictive_means)
        
    for cluster_id, cluster_center in tqdm(enumerate(cluster_centers)):
        if (cluster_assignments==cluster_id).sum()==0:
            # skip empty clusters
            continue

        train_features = feature_encoder.forward(train_data - cluster_center)

        assigned_test_data = test_data[cluster_assignments==cluster_id]
        if assigned_test_data.dim() == 1:
            assigned_test_data = assigned_test_data.unsqueeze(dim=0)
        test_features = feature_encoder.forward(assigned_test_data - cluster_center)

        het_gp = HeteroskedasticGP(None)

        f_test_mean, f_test_stds = het_gp.predictive_dist(
            train_features, test_features,
            train_labels, noise_var * torch.ones_like(train_labels)
        )

        predictive_means[cluster_assignments==cluster_id] = f_test_mean
        predictive_stds[cluster_assignments==cluster_id] = f_test_stds

    return predictive_means, predictive_stds