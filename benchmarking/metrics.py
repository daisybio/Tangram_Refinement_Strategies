import numpy as np
import torch
import scipy
import sklearn


# METRICS FOR GROUND TRUTH COMPARISONS

def cosine_similarity(true_values, pred_values, axis):
    """
    Compute the cosine similarity between true and predicted values
    Args:
        true_values (Array): Ground truth (k,j)
        pred_values (Array): Predicted values (r,k,j)
        axis (int): Axis where to perform the comparison, can be 1 or 2    
    Returns:
        Array: Cosine similarity values (r,k) o r(r,j)
    Example:
        Gene expression prediction correctness for each gene along the spots: n_runs x n_genes x n_spots => n_runs x n_genes
        Gene expression prediction correctness for each spot along the genes: n_runs x n_genes x n_spots => n_runs x n_spots
    """
    return np.array(torch.nn.functional.cosine_similarity(torch.Tensor(pred_values),
                                                          torch.Tensor(true_values),
                                                          dim=axis))

def categorical_cross_entropy(true_probs, pred_probs_cube):
    """
    Compute the categorical cross-entropy between true and predicted probabilities cube along the last axis
    Args:
        true_probs (Array): Ground truth (i,j)
        pred_probs_cube (Array): Predicted values (r,i,j)
    Returns:
        Array: Cross-entropy values (r,i)
    Example:
        Cell mapping correctness for each cell along the spots: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    entropy = []
    for run in range(pred_probs_cube.shape[0]):
        tmp = []
        for i in range(pred_probs_cube.shape[1]):
            tmp.append(sklearn.metrics.log_loss([true_probs[i].argmax()], np.array([pred_probs_cube[run,i]]), 
                                                labels=range(true_probs.shape[1]), normalize=False))
        entropy.append(tmp)
    return np.array(entropy) / np.array(entropy).mean(axis=0).max()

def multi_label_categorical_cross_entropy(true_probs, pred_probs_cube):
    """
    Compute the multi-label categorical cross-entropy between true and predicted probabilities cube along the last axis
    Args:
        true_probs (Array): Ground truth (i,j)
        pred_probs_cube (Array): Predicted values (r,i,j)
    Returns:
        Array: Multi-label cross-entropy values (r,i)
    Example:
        Cell type mapping correctness for each cell type along the spots: n_runs x n_celltypes x n_spots => n_runs x n_celltypes
    """
    entropy = []
    for run in range(pred_probs_cube.shape[0]):
        tmp = []
        for i in range(pred_probs_cube.shape[1]):
            tmp.append(sklearn.metrics.log_loss(np.array([true_probs[i], 1-true_probs[i]]).argmax(axis=0), 
                                                np.array([pred_probs_cube[run,i], 1-pred_probs_cube[run,i]]).T, 
                                                labels=[0,1], normalize=True))
        entropy.append(tmp)
    return np.array(entropy) / np.array(entropy).mean(axis=0).max()


# METRICS FOR TRAINING RUN COMPARISONS

def pearson_corr(cube):
    """
    Compute the pearson correlation for the first axis
    Args:
        cube (Array): Values (r,n,j)
    Returns:
        Array: All pairwise Pearson correlations (r x r)
    Example:
        Cell (type) mapping or gene expression prediction consistency: n_runs x n_genes/cell(type)s x n_spots => n_runpairs
    """
    idx = np.tril_indices(cube.shape[0], -1)
    return np.corrcoef(np.reshape(cube,(cube.shape[0],-1)))[idx]
    
def pearson_corr_over_axis(cube, axis):
    """
    Compute the pearson correlation for a given axis, averaged across all pairwise correlation from the first axis
    Args:
        cube (Array): Values (r,n,j)
        axis (int): Axis, can be 1 or 2
    Returns:
        Array: Mean Pearson correlations along a specific axis (n) or (j)
    Example:
        Cell (type) mapping or gene expression prediction consistency: n_runs x n_genes/cell(type)s x n_spots => n_genes/cell(type)s or n_spots
    """
    all_pearsons = []
    idx = np.tril_indices(cube.shape[0], -1)
    if axis == 1:
        for i in range(cube.shape[1]):
            all_pearsons.append(np.corrcoef(cube[:,i,:])[idx].mean())
    else: # axis == 2
        for i in range(cube.shape[2]):
            all_pearsons.append(np.corrcoef(cube[:,:,i])[idx].mean())
    return np.array(all_pearsons)

def vote_entropy(pred_probs_cube):
    """
    Compute the normalized vote entropy across the last axis
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Vote entropy values (r,i)
    Example:
        Cell mapping agreement: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    votes_encoded = np.zeros(pred_probs_cube.shape)
    votes = pred_probs_cube.argmax(axis=2)
    for run in range(pred_probs_cube.shape[0]):
        votes_encoded[run,np.arange(pred_probs_cube.shape[1]),votes[run]] = 1
    return scipy.stats.entropy(votes_encoded.mean(axis=0), axis=1) / np.log(pred_probs_cube.shape[2])

def multi_label_vote_entropy(pred_probs_cube):
    """
    Compute the normalized multi-label vote entropy
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Multi-label vote entropy values (r,i,j)
    Example:
        Cell type mapping agreement: n_runs x n_celltypes x n_spots => n_runs x n_celltypes x n_spots
    """
    votes_encoded = np.round(pred_probs_cube)
    return scipy.stats.entropy(np.array([votes_encoded.mean(axis=0), 1 - votes_encoded.mean(axis=0)]), axis=0) / np.log(2)

def consensus_entropy(pred_probs_cube):
    """
    Compute the normalized consensus entropy across the last axis
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Consensus entropy values (r,i)
    Example:
        Cell mapping certainty: n_runs x n_cells x n_spots => n_runs x n_cells
    """
    consensus_mapping = pred_probs_cube.mean(axis=0)
    return scipy.stats.entropy(consensus_mapping, axis=1) / np.log(pred_probs_cube.shape[2])

def multi_label_consensus_entropy(pred_probs_cube): 
    """
    Compute the normalized multi-label consensus entropy
    Args:
        pred_probs_cube (Array): Values (r,i,j)
    Returns:
        Array: Multi-label consensus entropy values (r,i,j)
    Example:
        Cell type mapping certainty: n_runs x n_celltypes x n_spots => n_runs x n_celltypes x n_spots
    """
    consensus_mapping = pred_probs_cube.mean(axis=0)
    return scipy.stats.entropy(np.array([consensus_mapping, 1 - consensus_mapping]), axis=0) / np.log(2)