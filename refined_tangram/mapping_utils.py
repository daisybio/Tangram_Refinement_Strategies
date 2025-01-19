"""
    Mapping helpers and hyperparameter tuning
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import logging

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from . import mapping_optimizer as mo
from . import utils as ut

#import ray
#from ray import train
#from ray import tune
#from ray.tune.search.optuna import OptunaSearch

import tangram_add_ons as tgx

logging.getLogger().setLevel(logging.INFO)


def pp_adatas(adata_sc, adata_sp, genes=None, gene_to_lowercase = True):
    """
    Pre-process AnnDatas so that they can be mapped. Specifically:
    - Remove genes that all entries are zero
    - Find the intersection between adata_sc, adata_sp and given marker gene list, save the intersected markers in two adatas
    - Calculate density priors and save it with adata_sp

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): spatial expression data
        genes (List): Optional. List of genes to use. If `None`, all genes are used.
    
    Returns:
        update adata_sc by creating `uns` `training_genes` `overlap_genes` fields 
        update adata_sp by creating `uns` `training_genes` `overlap_genes` fields and creating `obs` `rna_count_based_density` & `uniform_density` field
    """

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        # Use all genes
        genes = adata_sc.var.index
               
    # put all var index to lower case to align
    if gene_to_lowercase:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]
        adata_sp.var.index = [g.lower() for g in adata_sp.var.index]
        genes = list(g.lower() for g in genes)

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()
    

    # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(genes) & set(adata_sc.var.index) & set(adata_sp.var.index))
    # logging.info(f"{len(genes)} shared marker genes.")

    
    genes_train = genes
    adata_sc.uns["training_genes"] = genes_train
    adata_sp.uns["training_genes"] = genes_train
    logging.info(
        "{} training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.".format(
            len(genes_train)
        )
    )
    # Find overlap genes between two AnnDatas and sort them alphabetically (important for index based selection of validation genes)
    overlap_genes = np.sort(list(set(adata_sc.var.index) & set(adata_sp.var.index))).tolist()
    # logging.info(f"{len(overlap_genes)} shared genes.")

    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info(
        "{} overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.".format(
            len(overlap_genes)
        )
    )

    # Calculate uniform density prior as 1/number_of_spots
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info(
        f"uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
    )

    # Calculate rna_count_based density prior as % of rna molecule count
    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info(
        f"rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
    )
        

def adata_to_cluster_expression(adata, cluster_label, scale=True, add_density=True):
    """
    Convert an AnnData to a new AnnData with cluster expressions. Clusters are based on `cluster_label` in `adata.obs`.  The returned AnnData has an observation for each cluster, with the cluster-level expression equals to the average expression for that cluster.
    All annotations in `adata.obs` except `cluster_label` are discarded in the returned AnnData.
    
    Args:
        adata (AnnData): single cell data
        cluster_label (String): field in `adata.obs` used for aggregating values
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster. Default is True.
        add_density (bool): Optional. If True, the normalized number of cells in each cluster is added to the returned AnnData as obs.cluster_density. Default is True.

    Returns:
        AnnData: aggregated single cell data

    """
    try:
        value_counts = adata.obs[cluster_label].value_counts(normalize=True)
    except KeyError as e:
        raise ValueError("Provided label must belong to adata.obs.")
    unique_labels = value_counts.index
    new_obs = pd.DataFrame({cluster_label: unique_labels})
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)

    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, l in enumerate(unique_labels):
        if not scale:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.mean(axis=0)
        else:
            X_new[index] = adata[adata.obs[cluster_label] == l].X.sum(axis=0)
    adata_ret.X = X_new

    if add_density:
        adata_ret.obs["cluster_density"] = adata_ret.obs[cluster_label].map(
            lambda i: value_counts[i]
        )

    return adata_ret

def map_cells_to_space(
    adata_sc,
    adata_sp,
    cv_train_genes=None,
    train_genes_idx=None,
    val_genes_idx=None,
    cluster_label=None,
    mode="cells",
    device="cpu",
    learning_rate=0.1,
    num_epochs=1000,
    scale=True,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_l1=0,
    lambda_l2=0,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=None,
    lambda_sparsity_g1=0,
    lambda_neighborhood_g1=0,
    voxel_weights=None,
    lambda_ct_islands=0,
    neighborhood_filter=None,
    ct_encode=None,
    lambda_getis_ord=0,
    lambda_moran=0,
    lambda_geary=0,
    spatial_weights=None,
    random_state=None,
    verbose=True,
    density_prior='rna_count_based',
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_sp`).
    
    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cv_train_genes (list): Optional. Training gene list. Default is None.
        train_genes_idx (ndarray): Optional. Gene indices used for training from the training gene list.
        val_genes_idx (ndarray): Optional. Gene indices used for validation from the training gene list..
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for `mode=clusters`.
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'cell'.
        device (string or torch.device): Optional. Default is 'cpu'.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        scale (bool): Optional. Whether weight input single cell data by the number of cells in each cluster, only valid when cluster_label is not None. Default is True.
        lambda_d (float): Optional. Hyperparameter for the density term of the optimizer. Default is 0.
        lambda_g1 (float): Optional. Hyperparameter for the gene-voxel similarity term of the optimizer. Default is 1.
        lambda_g2 (float): Optional. Hyperparameter for the voxel-gene similarity term of the optimizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. An higher entropy promotes probabilities of each cell peaked over a narrow portion of space. lambda_r = 0 corresponds to no entropy regularizer. Default is 0.
        lambda_l1 (float): Optional. Strength of L1 regularizer. Default is 0.
        lambda_l2 (float): Optional. Strength of L2 regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        lambda_sparsity_g1 (float): Optional. Strength of sparsity weighted gene expression comparison. Default is 0.
        lambda_neighborhood_g1 (float): Optional. Strength of neighborhood weighted gene expression comparison. Default is 0.
        voxel_weights (ndarray): Optional. Spatial weight used for neighborhood weighting, shape = (number_spots, number_spots).
        lambda_getis_ord (float): Optional. Strength of Getis-Ord G* preservation. Default is 0.
        lambda_geary (float): Optional. Strength of Geary's C preservation. Default is 0.
        lambda_moran (float): Optional. Strength of Moran's I preservation. Default is 0.
        spatial_weights (ndarray): Optional. Spatial weight used for local spatial indicator preservation, shape = (number_spots, number_spots).
        lambda_ct_islands: Optional. Strength of ct islands enforcement. Default is 0.
        neighborhood_filter (ndarray): Optional. Neighborhood filter used for cell type island preservation, shape = (number_spots, number_spots).
        ct_encode(ndarray): Optional. One-hot encoding of cell types used for cell type island preservation, shape = (number_cells, number_celltypes).
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is True.
        density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If None, the density term is ignored. Default value is 'rna_count_based'.

    Returns:
        a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.
    """

    # check invalid values for arguments
    if lambda_g1+lambda_sparsity_g1 == 0:
        raise ValueError("lambda_g1 cannot be 0.")

    if (type(density_prior) is str) and (
        density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if density_prior is not None and (lambda_d == 0 or lambda_d is None):
        lambda_d = 1

    if lambda_d > 0 and density_prior is None:
        raise ValueError("When lambda_d is set, please define the density_prior.")

    if mode not in ["clusters", "cells", "constrained"]:
        raise ValueError('Argument "mode" must be "cells", "clusters" or "constrained')

    if mode == "clusters" and cluster_label is None:
        raise ValueError("A cluster_label must be specified if mode is 'clusters'.")

    if mode == "constrained" and not all([target_count, lambda_f_reg, lambda_count]):
        raise ValueError(
            "target_count, lambda_f_reg and lambda_count must be specified if mode is 'constrained'."
        )

    if mode == "clusters":
        adata_sc = adata_to_cluster_expression(
            adata_sc, cluster_label, scale, add_density=True
        )

    # Check if training_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get training_genes
    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    elif cv_train_genes is not None:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError(
                "Given training genes list should be subset of two AnnDatas."
            )

    logging.info("Allocate tensors for mapping.")
    # Allocate tensors (AnnData matrix can be sparse or not)

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    d_source = None

    # define density_prior if 'rna_count_based' is passed to the density_prior argument:
    d_str = density_prior
    if type(density_prior) is np.ndarray:
        d_str = "customized"

    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]

    # define density_prior if 'uniform' is passed to the density_prior argument:
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]

    if mode == "cells":
        d = density_prior

    if mode == "clusters":
        d_source = np.array(adata_sc.obs["cluster_density"])

    if mode in ["clusters", "constrained"]:
        if density_prior is None:
            d = adata_sp.obs["uniform_density"]
            d_str = "uniform"
        else:
            d = density_prior
        if lambda_d is None or lambda_d == 0:
            lambda_d = 1

    # Choose device
    device = torch.device(device)  # for gpu

    if verbose:
        print_each = 100
    else:
        print_each = None

    if mode in ["cells", "clusters"]:
        hyperparameters = {
            "lambda_d": lambda_d,
            "lambda_g1": lambda_g1,
            "lambda_g2": lambda_g2, 
            "lambda_r": lambda_r, 
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "d_source": d_source,
            "lambda_sparsity_g1": lambda_sparsity_g1,
            "lambda_neighborhood_g1": lambda_neighborhood_g1,
            "voxel_weights": voxel_weights,
            "lambda_ct_islands": lambda_ct_islands,
            "neighborhood_filter": neighborhood_filter,
            "ct_encode": ct_encode,
            "lambda_getis_ord": lambda_getis_ord,
            "lambda_moran": lambda_moran,
            "lambda_geary": lambda_geary,
            "spatial_weights": spatial_weights,
            "train_genes_idx": train_genes_idx,
            "val_genes_idx": val_genes_idx,
        }

        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )
        mapper = mo.Mapper(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters,
        )

        # TODO `train` should return the loss function

        mapping_matrix, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )

    # constrained mode
    elif mode == "constrained":
        hyperparameters = {
            "lambda_d": lambda_d,  # KL (ie density) term
            "lambda_g1": lambda_g1,  # gene-voxel cos sim
            "lambda_g2": lambda_g2,  # voxel-gene cos sim
            "lambda_r": lambda_r,  # regularizer: penalize entropy
            "lambda_count": lambda_count,
            "lambda_f_reg": lambda_f_reg,
            "target_count": target_count,
        }

        logging.info(
            "Begin training with {} genes and {} density_prior in {} mode...".format(
                len(training_genes), d_str, mode
            )
        )

        mapper = mo.MapperConstrained(
            S=S, G=G, d=d, device=device, random_state=random_state, **hyperparameters,
        )

        mapping_matrix, F_out, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )

    logging.info("Saving results..")
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    if mode == "constrained":
        adata_map.obs["F_out"] = F_out

    # Annotate cosine similarity of each training gene
    G_predicted = adata_map.X.T @ S
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    # Annotate sparsity of each training genes
    ut.annotate_gene_sparsity(adata_sc)
    ut.annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
        adata_sp[:, training_genes].var.sparsity
        - adata_sc[:, training_genes].var.sparsity
    )

    adata_map.uns["training_history"] = training_history

    return adata_map

def train_Mapper(config, data):
    """
    Wrapper function for hyperparameter tuning.
    config (dict): Hyperparameter setup.
    data (list): Needed data for training.
    """
    S,G,d_source,d,device,random_state,print_each,voxel_weights,ct_encode,neighborhood_filter,spatial_weights,train_genes_idx,val_genes_idx = data
    torch.manual_seed(random_state) 
    hyperparameters = {"d_source": d_source}
    for param in list(set(["lambda_d","lambda_g1","lambda_g2","lambda_neighborhood_g1","lambda_r","lambda_l1","lambda_l2","lambda_ct_islands","lambda_getis_ord"]).intersection(set(config.keys()))):
        hyperparameters[param] = config[param]

    mapper = mo.Mapper(
        S=S, 
        G=G, 
        d=d, 
        train_genes_idx=train_genes_idx, 
        val_genes_idx=val_genes_idx,
        voxel_weights=voxel_weights,
        neighborhood_filter=neighborhood_filter,
        ct_encode=ct_encode,
        spatial_weights=spatial_weights,
        device=device, 
        random_state=random_state, 
        **hyperparameters,
    )
    learning_rate = 0.1
    if "learning_rate" in config.keys():
        learning_rate = config["learning_rate"]
    num_epochs = 1000
    if "num_epochs" in config.keys():
        num_epochs = config["num_epochs"]
    _, training_history = mapper.train(
        print_each=print_each, 
        val_each=1000,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )
    train.report({"val_gene_score" : training_history["val_gene_score"][-1],
                  "val_sp_sparsity_weighted_score" : training_history["val_sp_sparsity_weighted_score"][-1],
                  "val_auc_score" : training_history["val_auc_score"][-1],
                  "val_prob_entropy" : training_history["val_prob_entropy"][-1]})

def train_multiple_Mapper(config,data):
    """
    Wrapper function for hyperparameter tuning, enables to evaluate consistency meassurements by training multiple mappers for each configuration.
    config (dict): Hyperparameter setup.
    data (list): Needed data for training.
    """
    S,G,d_source,d,device,random_state,print_each,voxel_weights,ct_encode,neighborhood_filter,spatial_weights,train_genes_idx,val_genes_idx = data
    torch.manual_seed(random_state) 
    hyperparameters = {"d_source": d_source}
    for param in list(set(["lambda_d","lambda_g1","lambda_g2","lambda_neighborhood_g1","lambda_r","lambda_l1","lambda_l2","lambda_ct_islands","lambda_getis_ord"]).intersection(set(config.keys()))):
        hyperparameters[param] = config[param]
  
    learning_rate = 0.1
    if "learning_rate" in config.keys():
        learning_rate = config["learning_rate"]
    num_epochs = 1000
    if "num_epochs" in config.keys():
        num_epochs = config["num_epochs"]

    mapping_matrices = list()
    val_gene_scores = list()
    val_sp_sparsity_weighted_scores = list()
    val_auc_scores = list()
    val_prob_entropies = list()
    for run in range(3):
        mapper = mo.Mapper(
            S=S, 
            G=G, 
            d=d, 
            train_genes_idx=train_genes_idx, 
            val_genes_idx=val_genes_idx,
            voxel_weights=voxel_weights,
            neighborhood_filter=neighborhood_filter,
            ct_encode=ct_encode,
            spatial_weights=spatial_weights,
            device=device, 
            random_state=run, 
            **hyperparameters,
        )
        mapping_matrix, training_history = mapper.train(
            print_each=print_each, 
            validate=True,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
        mapping_matrices.append(mapping_matrix)
        val_gene_scores.append(training_history["val_gene_score"][-1])
        val_sp_sparsity_weighted_scores.append(training_history["val_sp_sparsity_weighted_score"][-1])
        val_auc_scores.append(training_history["val_auc_score"][-1])
        val_prob_entropies.append(training_history["val_prob_entropy"][-1])

    cell_mapping_cube = np.array(mapping_matrices)    
    gene_expr_cube = np.array([(S[:,val_genes_idx].T @ mapping_matrix) for mapping_matrix in mapping_matrices])
    train.report({"cell_map_corr" : tgx.pearson_corr(cell_mapping_cube).mean(),
                  "cell_map_vote_entropy" : tgx.vote_entropy(cell_mapping_cube).mean(),
                  "cell_map_consensus_entropy" : tgx.consensus_entropy(cell_mapping_cube).mean(),
                  "gene_corr" : tgx.pearson_corr(gene_expr_cube).mean(),
                  "val_gene_score" : np.array(val_gene_scores).mean(),
                  "val_sp_sparsity_weighted_score" : np.array(val_sp_sparsity_weighted_scores).mean(),
                  "val_auc_score" : np.array(val_auc_scores).mean(),
                  "val_prob_entropy" : np.array(val_prob_entropies).mean()})

def map_cells_to_space_hyperparameter_tuning(
    adata_sc,
    adata_sp,
    cv_train_genes=None,
    train_genes_idx=None,
    val_genes_idx=None,
    mode="cells",
    device="cpu",
    random_state=None,
    density_prior='rna_count_based',
    voxel_weights=None,
    ct_encode=None,
    neighborhood_filter=None,
    spatial_weights=None,
    config=None,
):
    if (type(density_prior) is str) and (
        density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if mode not in ["cells"]:
        raise ValueError('Argument "mode" must be "cells"')

    # Check if training_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get training_genes
    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    elif cv_train_genes is not None:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError(
                "Given training genes list should be subset of two AnnDatas."
            )

    logging.info("Allocate tensors for mapping.")
    # Allocate tensors (AnnData matrix can be sparse or not)

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    d_source = None

    # define density_prior if 'rna_count_based' is passed to the density_prior argument:
    d_str = density_prior
    if type(density_prior) is np.ndarray:
        d_str = "customized"

    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]

    # define density_prior if 'uniform' is passed to the density_prior argument:
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]

    if mode == "cells":
        d = density_prior

    # Choose device
    device = torch.device(device)  # for gpu

    print_each = None

    if mode in ["cells"]:
        ray.init(address = "auto",_temp_dir='/nfs/home/students/m.stahl/ray/')
        data = [S,G,d_source,d,device,random_state,print_each,voxel_weights,ct_encode,neighborhood_filter,spatial_weights,train_genes_idx,val_genes_idx]

        optuna_search = OptunaSearch(
            metric=["val_gene_score","val_auc_score","val_sp_sparsity_weighted_score","val_prob_entropy"],
            mode=["max","max","max","min"])

        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(train_Mapper,data=data), {"gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                num_samples=1,
            ),
            param_space=config,
        )
        tuner.fit()
        return tuner
    
def map_cells_to_space_hyperparameter_grid_search(
    adata_sc,
    adata_sp,
    cv_train_genes=None,
    train_genes_idx=None,
    val_genes_idx=None,
    mode="cells",
    device="cpu",
    random_state=None,
    verbose=True,
    density_prior='rna_count_based',
    ct_encode=None,
    neighborhood_filter=None,
    config=None,
):

    if (type(density_prior) is str) and (
        density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if mode not in ["cells"]:
        raise ValueError('Argument "mode" must be "cells"')

    # Check if training_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get training_genes
    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    elif cv_train_genes is not None:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError(
                "Given training genes list should be subset of two AnnDatas."
            )

    logging.info("Allocate tensors for mapping.")
    # Allocate tensors (AnnData matrix can be sparse or not)

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    d_source = None
    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]

    # define density_prior if 'uniform' is passed to the density_prior argument:
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]

    if mode == "cells":
        d = density_prior

    # Choose device
    device = torch.device(device)  # for gpu

    if verbose:
        print_each = 10
    else:
        print_each = None

    if mode in ["cells"]:
        ray.init(address = "auto",_temp_dir='/nfs/home/students/m.stahl/ray/')
        data = [S,G,d_source,d,device,random_state,print_each,ct_encode,neighborhood_filter,train_genes_idx,val_genes_idx]
        origopt = ray.tune.search.basic_variant.BasicVariantGenerator(constant_grid_search=True)
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(train_Mapper,data=data), {"gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=origopt,
            ),
            param_space=config,
        )
        tuner.fit()
        return tuner

def map_cells_to_space_hyperparameter_tuning_consistency(
    adata_sc,
    adata_sp,
    cv_train_genes=None,
    train_genes_idx=None,
    val_genes_idx=None,
    mode="cells",
    device="cpu",
    random_state=None,
    density_prior='rna_count_based',
    voxel_weights=None,
    ct_encode=None,
    neighborhood_filter=None,
    spatial_weights=None,
    config=None,
):
    if (type(density_prior) is str) and (
        density_prior not in ["rna_count_based", "uniform", None]
    ):
        raise ValueError("Invalid input for density_prior.")

    if mode not in ["cells"]:
        raise ValueError('Argument "mode" must be "cells"')

    # Check if training_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get training_genes
    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    elif cv_train_genes is not None:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError(
                "Given training genes list should be subset of two AnnDatas."
            )

    logging.info("Allocate tensors for mapping.")
    # Allocate tensors (AnnData matrix can be sparse or not)

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32",)
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    d_source = None

    if density_prior == "rna_count_based":
        density_prior = adata_sp.obs["rna_count_based_density"]

    # define density_prior if 'uniform' is passed to the density_prior argument:
    elif density_prior == "uniform":
        density_prior = adata_sp.obs["uniform_density"]

    if mode == "cells":
        d = density_prior

    # Choose device
    device = torch.device(device)  # for gpu

    print_each = None

    if mode in ["cells"]:
        ray.init(address = "auto",_temp_dir='/nfs/home/students/m.stahl/ray/')

        data = [S,G,d_source,d,device,random_state,print_each,voxel_weights,ct_encode,neighborhood_filter,spatial_weights,train_genes_idx,val_genes_idx]

        optuna_search = OptunaSearch(
            metric=["val_gene_score",
                    "cell_map_corr",
                    "cell_map_vote_entropy",
                    "cell_map_consensus_entropy",
                    ],
            mode=["max",
                  "max",
                  "min",
                  "min",
                  ])

        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(train_multiple_Mapper,data=data), {"gpu": 1}),
            tune_config=tune.TuneConfig(
                search_alg=optuna_search,
                num_samples=2000,
            ),
            param_space=config,
        )
        tuner.fit()
        return tuner


