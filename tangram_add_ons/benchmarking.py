import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import metrics as m

sys.path.insert(0,'../')
import refined_tangram as tg


def cell_type_mapping(adata_map, cell_types_key="cell_types"):
    """
    Compute the cell type mapping based on the cell mapping
    Args:
        adata_maps_pred (AnnData): Mapping data
        cell_types_key (str): Optional. Key for cell type labels
    
    Returns:
        None.
        Update mapping AnnData by creating `varm` `ct_map` field which contains a dataframe with the cell type mapping
    
    """
    df = tg.one_hot_encoding(adata_map.obs[cell_types_key])
    df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index
    vmin = df_ct_prob.min() 
    vmax = df_ct_prob.max()
    df_ct_prob = (df_ct_prob - vmin) / (vmax - vmin) # per cell type normalization
    adata_map.varm["ct_map"] = df_ct_prob


def benchmark_metrics(adata_maps_pred, adata_sc, adata_st, adata_maps_true=None):
    metrics = {
        "gene_expr_correctness" : dict(),
        "gene_expr_consistency" : dict(),
        "cell_map_consistency" : dict(),
        "cell_map_agreement" : dict(),
        "cell_map_certainty" : dict(),
        "ct_map_consistency" : dict(),
        "ct_map_agreement" : dict(),
        "ct_map_certainty" : dict()
    }
    if adata_maps_true is not None:
        metrics["cell_map_correctness"] = dict()
        metrics["ct_map_correctness"] = dict()

    test_genes = adata_sc.uns["test_genes"]
    true_gene_expr = adata_st[:,test_genes].X.T.copy()

    for model in tqdm(adata_maps_pred.keys()):

        cell_mapping_cube = np.array([adata_maps_pred[model][run].X for run in adata_maps_pred[model].keys()])
        metrics["cell_map_consistency"][model] = m.pearson_corr_over_axis(cell_mapping_cube, axis=1)
        metrics["cell_map_agreement"][model] = m.vote_entropy(cell_mapping_cube)
        metrics["cell_map_certainty"][model] = m.consensus_entropy(cell_mapping_cube)
        if adata_maps_true is not None:
            metrics["cell_map_correctness"][model] = m.categorical_cross_entropy(adata_maps_true.X, cell_mapping_cube)

        celltype_mapping_cube = np.array([adata_maps_pred[model][run].varm["ct_map"].values.T for run in adata_maps_pred[model].keys()])
        metrics["ct_map_consistency"][model] = m.pearson_corr_over_axis(celltype_mapping_cube, axis=1)
        metrics["ct_map_agreement"][model] =  m.multi_label_vote_entropy(celltype_mapping_cube).tolist()
        metrics["ct_map_certainty"][model] = m.multi_label_consensus_entropy(celltype_mapping_cube).tolist()
        if adata_maps_true is not None:
            metrics["ct_map_correctness"][model] = m.multi_label_categorical_cross_entropy(adata_maps_true.varm["ct_map"], celltype_mapping_cube)
 
        gene_expr_cube = np.array([(adata_sc[:,test_genes].X.T @ adata_maps_pred[model][run].X) for run in adata_maps_pred[model].keys()])
        metrics["gene_expr_correctness"][model] = m.cosine_similarity(true_gene_expr, gene_expr_cube, 2).tolist()
        metrics["gene_expr_consistency"][model] = m.pearson_corr_over_axis(gene_expr_cube, axis=1)
    
    return metrics

def benchmark_metrics_constrained(adata_maps_pred, adata_sc, adata_st, adata_maps_true=None):
    metrics = {
        "gene_expr_correctness" : dict(),
        "gene_expr_consistency" : dict(),
        "cell_map_consistency" : dict(),
        "cell_map_agreement" : dict(),
        "cell_map_certainty" : dict(),
        "ct_map_consistency" : dict(),
        "ct_map_agreement" : dict(),
        "ct_map_certainty" : dict()
    }
    if adata_maps_true is not None:
        metrics["cell_map_correctness"] = dict()
        metrics["ct_map_correctness"] = dict()

    test_genes = adata_sc.uns["test_genes"]
    true_gene_expr = adata_st[:,test_genes].X.T.copy()

    for model in tqdm(adata_maps_pred.keys()):

        cell_mapping_cube = np.array([np.array([adata_maps_pred[model][run].obs["F_out"]]).T * adata_maps_pred[model][run].X for run in adata_maps_pred[model].keys()])
        metrics["cell_map_consistency"][model] = m.pearson_corr_over_axis(cell_mapping_cube, axis=1)
        metrics["cell_map_agreement"][model] = m.vote_entropy(cell_mapping_cube)
        metrics["cell_map_certainty"][model] = m.consensus_entropy(cell_mapping_cube)
        if adata_maps_true is not None:
            metrics["cell_map_correctness"][model] = m.categorical_cross_entropy(adata_maps_true.X, cell_mapping_cube)

        celltype_mapping_cube = np.array([adata_maps_pred[model][run].varm["ct_map"].values.T for run in adata_maps_pred[model].keys()])
        metrics["ct_map_consistency"][model] = m.pearson_corr_over_axis(celltype_mapping_cube, axis=1)
        metrics["ct_map_agreement"][model] =  m.multi_label_vote_entropy(celltype_mapping_cube).tolist()
        metrics["ct_map_certainty"][model] = m.multi_label_consensus_entropy(celltype_mapping_cube).tolist()
        if adata_maps_true is not None:
            metrics["ct_map_correctness"][model] = m.multi_label_categorical_cross_entropy(adata_maps_true.varm["ct_map"], celltype_mapping_cube)
 
        gene_expr_cube = np.array([((np.array([adata_maps_pred[model][run].obs["F_out"]]).T * adata_sc[:,test_genes].X).T @ adata_maps_pred[model][run].X) for run in adata_maps_pred[model].keys()])
        metrics["gene_expr_correctness"][model] = m.cosine_similarity(true_gene_expr, gene_expr_cube, 2).tolist()
        metrics["gene_expr_consistency"][model] = m.pearson_corr_over_axis(gene_expr_cube, axis=1)
    
    return metrics

def benchmark_detailed_metrics(adata_maps_pred, adata_sc, adata_st):
    metrics = {
        "gene_expr_correctness_gene" : dict(),
        "gene_expr_correctness_spot" : dict(),
        "gene_expr_consistency_gene" : dict(),
        "gene_expr_consistency_spot" : dict(),
        "cell_map_consistency_cell" : dict(),
        "cell_map_consistency_spot" : dict(),
        "cell_map_agreement_cell" : dict(),
        "cell_map_agreement_spot" : dict(),
        "cell_map_certainty_cell" : dict(),
        "cell_map_certainty_spot" : dict(),
        "ct_map_consistency_ct" : dict(),
        "ct_map_consistency_spot" : dict(),
        "ct_map_agreement" : dict(),
        "ct_map_certainty" : dict()
    }

    test_genes = adata_sc.uns["test_genes"]
    true_gene_expr = adata_st[:,test_genes].X.T.copy()

    for model in tqdm(adata_maps_pred.keys()):

        cell_mapping_cube = np.array([adata_maps_pred[model][run].X for run in adata_maps_pred[model].keys()])
        metrics["cell_map_consistency_cell"][model] = m.pearson_corr_over_axis(cell_mapping_cube, axis=1)
        metrics["cell_map_consistency_spot"][model] = m.pearson_corr_over_axis(cell_mapping_cube, axis=2)
        metrics["cell_map_agreement_cell"][model] = m.vote_entropy(cell_mapping_cube)
        metrics["cell_map_agreement_spot"][model] = m.multi_label_vote_entropy(cell_mapping_cube)
        metrics["cell_map_certainty_cell"][model] = m.consensus_entropy(cell_mapping_cube)
        metrics["cell_map_certainty_spot"][model] = m.multi_label_consensus_entropy(cell_mapping_cube)

        celltype_mapping_cube = np.array([adata_maps_pred[model][run].varm["ct_map"].values.T for run in adata_maps_pred[model].keys()])
        metrics["ct_map_consistency_ct"][model] = m.pearson_corr_over_axis(celltype_mapping_cube, axis=1)
        metrics["ct_map_consistency_spot"][model] = m.pearson_corr_over_axis(celltype_mapping_cube, axis=2)
        metrics["ct_map_agreement"][model] =  m.multi_label_vote_entropy(celltype_mapping_cube).tolist()
        metrics["ct_map_certainty"][model] = m.multi_label_consensus_entropy(celltype_mapping_cube).tolist()
    
        gene_expr_cube = np.array([(adata_sc[:,test_genes].X.T @ adata_maps_pred[model][run].X) for run in adata_maps_pred[model].keys()])
        metrics["gene_expr_correctness_gene"][model] = m.cosine_similarity(true_gene_expr, gene_expr_cube, 2).tolist()
        metrics["gene_expr_correctness_spot"][model] = m.cosine_similarity(true_gene_expr, gene_expr_cube, 1).tolist()
        metrics["gene_expr_consistency_gene"][model] = m.pearson_corr_over_axis(gene_expr_cube, axis=1)
        metrics["gene_expr_consistency_spot"][model] = m.pearson_corr_over_axis(gene_expr_cube, axis=2)
    
    return metrics

def mean_metrics(metrics):
    return pd.DataFrame({metric : {model : np.mean(metrics[metric][model]) for model in metrics[metric].keys()} for metric in metrics.keys()})
