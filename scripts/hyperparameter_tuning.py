#!/usr/bin/env python3

import squidpy as sq
import numpy as np
import tangram_modified_version as tg
import ray
from ray import tune

# load data
adata_sc = sq.datasets.sc_mouse_cortex()
adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[
    adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
].copy()

# pre-processing
with open('../data/spapros_genes.txt') as f:
    genes = eval(f.readline())
tg.pp_adatas(adata_sc, adata_st, genes=genes)

# hyperparameter tuning
metric = ["cell_map_consistency",
          "cell_map_agreement",
          "cell_map_certainty",
          "gene_expr_correctness"]

config = {
        "learning_rate" : tune.loguniform(0.001, 1),
        "lambda_g1": tune.uniform(0, 1.0),
        "lambda_g2": tune.uniform(0, 1.0),
        "lambda_d": tune.uniform(0, 1.0),
        "lambda_r": tune.loguniform(1e-20, 1e-3),
        "lambda_l2": tune.loguniform(1e-20, 1e-3),
        "lambda_neighborhood_g1": tune.uniform(0, 1.0),
        "lambda_ct_islands": tune.uniform(0, 1.0),
        "lambda_getis_ord": tune.uniform(0, 1.0),
}

tuner = tg.map_cells_to_space_hyperparameter_tuning(
        adata_sc, 
        adata_st,
        metric,
        config,
        mode="cells",
        density_prior='rna_count_based',
        device='cuda:0',
        random_state=1234
    )  
df = tuner.get_results().get_dataframe()
df.to_csv("../hyperparameter_tuning.csv")

ray.shutdown()
    