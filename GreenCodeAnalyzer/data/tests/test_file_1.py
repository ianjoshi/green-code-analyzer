import torch
import numpy as np



def fmes_adjustment(gradients, overlap_weight_index):
    params_to_add = []
    for client_id in range(len(gradients)):
        if overlap_weight_index[client_id] == 2:
            params_to_add.append(gradients[client_id])
        elif overlap_weight_index[client_id] == 3:
            params_to_add.append(gradients[client_id])
            params_to_add.append(gradients[client_id])

    stacked_params_to_add = torch.stack(params_to_add, dim=0)
    return torch.cat((gradients, stacked_params_to_add), dim=0)


# In fedmes paper 'https://ieeexplore-ieee-org.tudelft.idm.oclc.org/document/9562553/metrics#metrics' it's really
# described more as a mean. So rather not use this
def fmes_median(gradients, overlap_weight_index):
    adjusted_gradients = fmes_adjustment(gradients, overlap_weight_index)
    return torch.median(adjusted_gradients, dim=0)[0]


def fmes_mean(gradients, overlap_weight_index):
    adjusted_gradients = fmes_adjustment(gradients, overlap_weight_index)
    return torch.mean(adjusted_gradients, dim=0)


# def fedmes_tr_mean_v1(all_updates, n_attackers, overlap_weight_index):
#     adjusted_gradients = fedmes_adjustment_selected(all_updates, overlap_weight_index)
#     sorted_updates = torch.sort(adjusted_gradients, 0)[0]
#     out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates, 0)
#     return out


def fmes_tr_mean(all_updates, n_attackers, overlap_weight_index):
    sorted_updates, sorted_indexes = torch.sort(all_updates, 0)
    selected_indexes = sorted_indexes[n_attackers:-n_attackers]
    selected_updates = sorted_updates[n_attackers:-n_attackers]

    return fmes_elementwise_mean(selected_indexes, selected_updates, overlap_weight_index)


# def fedmes_elementwise_mean(selected_indexes, selected_updates, overlap_weight_index):
#     print(selected_indexes)
#     total_selected_weight = torch.zeros_like(selected_indexes[0], dtype=torch.float32)

#     # adjust for overlapping regions
#     fedmes_mean_selected = torch.zeros_like(selected_indexes[0], dtype=torch.float32)
#     for i in range(len(selected_indexes)):
#         print (len(selected_indexes[i]))
#         for j in range(len(selected_indexes[i])):
#             client_index = selected_indexes[i][j].item()
#             fedmes_mean_selected[j] += selected_updates[i][j] * overlap_weight_index[client_index]
#             total_selected_weight[j] += overlap_weight_index[client_index]

#     return fedmes_mean_selected / total_selected_weight

def fmes_elementwise_mean(selected_indexes, selected_updates, overlap_weight_index):
    total_selected_weight = torch.zeros_like(selected_indexes[0], dtype=torch.float32)

    # Adjust for overlapping regions
    fedmes_mean_selected = torch.zeros_like(selected_indexes[0], dtype=torch.float32)
    
    overlap_weight_tensor= torch.tensor(overlap_weight_index).cuda()
    
    for i in range(len(selected_indexes)):
        client_indices = selected_indexes[i].long()  # Ensure indices are of type long
        client_weights = overlap_weight_tensor[client_indices]
        
        fedmes_mean_selected += selected_updates[i] * client_weights
        total_selected_weight += client_weights

    # Use broadcasting to divide element-wise
    result = fedmes_mean_selected / total_selected_weight

    return result


def fmes_adjustment_selected(candidates, candidate_indices, overlap_weight_index):
    params_to_add = []
    for client_id in range(len(candidates)):
        if overlap_weight_index[candidate_indices[client_id]] == 2:
            params_to_add.append(candidates[client_id])
        elif overlap_weight_index[candidate_indices[client_id]] == 3:
            params_to_add.append(candidates[client_id])
            params_to_add.append(candidates[client_id])
    if len(params_to_add) > 1:
        stacked_params_to_add = torch.stack(params_to_add, dim=0)
    else:
        return candidates
    return torch.cat((candidates, stacked_params_to_add), dim=0)


def fmes_multi_krum(all_updates, n_attackers, overlap_weight_index, multi_k=False):
    candidates = []
    candidate_indices = []
    
    # If doing krum we have to add the fedmes bias before since only one candidate is selected.
    if not multi_k:
        all_updates = fmes_adjustment(all_updates, overlap_weight_index)
    
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        # torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat(
            (candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # Adjust for fedmes
    fedmes_adjusted_candidates = fmes_adjustment_selected(candidates, candidate_indices, overlap_weight_index)

    aggregate = torch.mean(fedmes_adjusted_candidates, dim=0)

    return aggregate


def fmes_bulyan(all_updates, n_attackers, overlap_weight_index):
    nusers = all_updates.shape[0]
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        # torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        # print(distances)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat(
            (bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    # Get index of params from sorted_params in original all_updates list
    og_indexes = []
    for update in sort_idx:
        og_indexes.append(candidate_indices[update[0].item()])

    # Get selected_updates and their indexes
    selected_indexes = og_indexes[:n - n_attackers]
    selected_updates = sorted_params[:n - n_attackers]

    # Adjust for fedmes
    fedmes_adjusted_candidates = fmes_adjustment_selected(selected_updates, selected_indexes, overlap_weight_index)

    aggregate = torch.mean(fedmes_adjusted_candidates, dim=0)

    return aggregate
   
def fmes_dnc(all_updates, n_attackers, overlap_weight_index, filtering_fraction, niters, subsample_dimension):
    selected_indices_sets = []

    gradient_dimension = all_updates.shape[1]
    
    for _ in range(niters):
        # Randomly select subsample_dimension number of dimensions and afterwards sort them once again
        random_dimensions = torch.sort(torch.randperm(gradient_dimension)[:subsample_dimension]).values
        
        # Subsample the input gradients using the randomly selected dimensions
        subsampled_gradients = all_updates[:, random_dimensions]
        
        # Compute the mean of the subsampled gradients along each dimension
        mean_gradients = torch.mean(subsampled_gradients, dim=0)
        
        # Center the subsampled gradients by subtracting the mean
        centered_gradients = subsampled_gradients - mean_gradients

        # Perform Singular Value Decomposition (SVD) to get the top right singular vector
        _, _, right_singular_vector = torch.svd(centered_gradients)
        
        # The outlier scores are the values of the variance squared and variance is given by the top right singular value
        top_right_singular_vector = right_singular_vector[:, 0]
        
        projection = torch.matmul(subsampled_gradients, top_right_singular_vector)
        
        scores = torch.square(projection)
        
        

        # Sort in ascending order and take indices of updates with least variance
        last_index = len(all_updates) - int(filtering_fraction * n_attackers)
        _, sorted_indices = torch.sort(scores)
        selected_indices = sorted_indices[:last_index]
        
        
        # Compute indices of the gradients with lowest outlier scores
        selected_indices_sets.append(set(selected_indices.cpu().numpy()))

    # Get intersection of all selected indices sets
    final_indices_set = set.intersection(*selected_indices_sets)
    
    # Get selected_updates
    final_selected_indices = [i for i in final_indices_set]
    selected_updates = all_updates[final_selected_indices]

    # Adjust for fedmes
    fedmes_adjusted_candidates = fmes_adjustment_selected(selected_updates, final_selected_indices, overlap_weight_index)

    aggregate = torch.mean(fedmes_adjusted_candidates, dim=0)

    return aggregate

def ms_dnc(all_updates, n_attackers, overlap_weight_index, filtering_fraction, niters, subsample_dimension):
    selected_indices_sets = []

    gradient_dimension = all_updates.shape[1]
    
    for _ in range(niters):
        # Randomly select subsample_dimension number of dimensions and afterwards sort them once again
        random_dimensions = torch.sort(torch.randperm(gradient_dimension)[:subsample_dimension]).values
        
        # Subsample the input gradients using the randomly selected dimensions
        subsampled_gradients = all_updates[:, random_dimensions]
        
        # Compute the mean of the subsampled gradients along each dimension
        mean_gradients = torch.mean(subsampled_gradients, dim=0)
        
        # Center the subsampled gradients by subtracting the mean
        centered_gradients = subsampled_gradients - mean_gradients

        # Perform Singular Value Decomposition (SVD) to get the top right singular vector
        _, _, right_singular_vector = torch.svd(centered_gradients)
        
        # The outlier scores are the values of the variance squared and variance is given by the top right singular value
        principle_component = right_singular_vector[:, 0]
        
        projection = torch.matmul(subsampled_gradients, principle_component)

        
        scores = torch.square(projection)
        
        

        # Sort in ascending order and take indices of updates with least variance
        last_index = len(all_updates) - int(filtering_fraction * n_attackers)
        _, sorted_indices = torch.sort(scores)
        selected_indices = sorted_indices[:last_index]
        
        
        # Compute indices of the gradients with lowest outlier scores
        selected_indices_sets.append(set(selected_indices.cpu().numpy()))

    # Get intersection of all selected indices sets
    final_indices_set = set.intersection(*selected_indices_sets)
    
    # Get selected_updates
    final_selected_indices = [i for i in final_indices_set]
    selected_updates = all_updates[final_selected_indices]

    aggregate = torch.mean(selected_updates, dim=0)

    return aggregate