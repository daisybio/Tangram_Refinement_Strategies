import sys
import numpy as np
from tqdm import tqdm

from . import metrics as m

sys.path.insert(0,'../')

def eval_metrics(adata_maps_pred, adata_sc, adata_st, adata_maps_true=None):
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
        metrics["cell_map_agreement"][model] = 1-m.vote_entropy(cell_mapping_cube)
        metrics["cell_map_certainty"][model] = 1-m.consensus_entropy(cell_mapping_cube)
        if adata_maps_true is not None:
            metrics["cell_map_correctness"][model] = 1-m.categorical_cross_entropy(adata_maps_true.X, cell_mapping_cube)

        celltype_mapping_cube = np.array([adata_maps_pred[model][run].varm["ct_map"].values.T for run in adata_maps_pred[model].keys()])
        metrics["ct_map_consistency"][model] = m.pearson_corr_over_axis(celltype_mapping_cube, axis=1)
        metrics["ct_map_agreement"][model] =  1-m.multi_label_vote_entropy(celltype_mapping_cube).tolist()
        metrics["ct_map_certainty"][model] = 1-m.multi_label_consensus_entropy(celltype_mapping_cube).tolist()
        if adata_maps_true is not None:
            metrics["ct_map_correctness"][model] = 1-m.multi_label_categorical_cross_entropy(adata_maps_true.varm["ct_map"], celltype_mapping_cube)
 
        gene_expr_cube = np.array([(adata_sc[:,test_genes].X.T @ adata_maps_pred[model][run].X) for run in adata_maps_pred[model].keys()])
        metrics["gene_expr_correctness"][model] = m.cosine_similarity(true_gene_expr, gene_expr_cube, 2).tolist()
        metrics["gene_expr_consistency"][model] = m.pearson_corr_over_axis(gene_expr_cube, axis=1)
    
    return metrics

def eval_metrics_constrained(adata_maps_pred, adata_sc, adata_st, adata_maps_true=None):
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
        metrics["cell_map_agreement"][model] = 1-m.vote_entropy(cell_mapping_cube)
        metrics["cell_map_certainty"][model] = 1-m.consensus_entropy(cell_mapping_cube)
        if adata_maps_true is not None:
            metrics["cell_map_correctness"][model] = 1-m.categorical_cross_entropy(adata_maps_true.X, cell_mapping_cube)

        celltype_mapping_cube = np.array([adata_maps_pred[model][run].varm["ct_map"].values.T for run in adata_maps_pred[model].keys()])
        metrics["ct_map_consistency"][model] = m.pearson_corr_over_axis(celltype_mapping_cube, axis=1)
        metrics["ct_map_agreement"][model] =  1-m.multi_label_vote_entropy(celltype_mapping_cube).tolist()
        metrics["ct_map_certainty"][model] = 1-m.multi_label_consensus_entropy(celltype_mapping_cube).tolist()
        if adata_maps_true is not None:
            metrics["ct_map_correctness"][model] = 1-m.multi_label_categorical_cross_entropy(adata_maps_true.varm["ct_map"], celltype_mapping_cube)
 
        gene_expr_cube = np.array([((np.array([adata_maps_pred[model][run].obs["F_out"]]).T * adata_sc[:,test_genes].X).T @ adata_maps_pred[model][run].X) for run in adata_maps_pred[model].keys()])
        metrics["gene_expr_correctness"][model] = m.cosine_similarity(true_gene_expr, gene_expr_cube, 2).tolist()
        metrics["gene_expr_consistency"][model] = m.pearson_corr_over_axis(gene_expr_cube, axis=1)
    
    return metrics

def mean_metrics(metrics, axis=None):
    return {metric : {model : np.mean(metrics[metric][model], axis=axis) for model in metrics[metric].keys()} for metric in metrics.keys()}
