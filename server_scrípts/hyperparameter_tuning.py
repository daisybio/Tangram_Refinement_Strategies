#!/usr/bin/env python3
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import tangram_modified_version as tg
import benchmarking as tgx
import anndata as an
import ray
from ray import tune

def main():
    # load data
    adata_sc = sq.datasets.sc_mouse_cortex()
    adata_st = sq.datasets.visium_fluo_adata_crop()
    adata_st = adata_st[
        adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
    ].copy()

    # pre-processing
    tg.pp_adatas(adata_sc, adata_st)
    adata_sc.uns["overlap_genes"] = np.sort(adata_sc.uns["overlap_genes"]).tolist()

    sq.gr.spatial_neighbors(adata_st, set_diag = False)
    voxel_weights = tgx.spatial_weights(adata_st, diag_zero = False)
    neighborhood_filter = np.array(adata_st.obsp["spatial_connectivities"].todense())
    ct_encode = tg.one_hot_encoding(adata_sc.obs["cell_subclass"]).values
    spatial_weights = tgx.getis_ord_G_star_spatial_weights(adata_st)

    config = {
            "learning_rate" : tune.loguniform(0.001, 1),#tune.loguniform(0.01, 1),
            "lambda_g1": tune.uniform(0, 1.0),
            "lambda_g2": tune.uniform(0, 1.0),
            "lambda_d": tune.uniform(0, 1.0),
            "lambda_r": tune.loguniform(1e-20, 1e-3),
            "lambda_l2": tune.loguniform(1e-20, 1e-3),
            "lambda_neighborhood_g1": tune.uniform(0, 1.0),
            "lambda_ct_islands": tune.uniform(0, 1.0),
            "lambda_getis_ord": tune.uniform(0, 1.0),

    }
    tuner = tg.map_cells_to_space_hyperparameter_tuning_multiple(
            adata_sc, 
            adata_st,
            mode="cells",
            density_prior='rna_count_based',
            voxel_weights=voxel_weights,
            neighborhood_filter=neighborhood_filter,
            ct_encode=ct_encode,
            spatial_weights=spatial_weights,
            device='cuda:0',
            random_state=1234,
            config=config
        )  
    df = tuner.get_results().get_dataframe()
    df.to_csv("hyperparameter/hyperparam_tuning_Spapros.csv")

    ray.shutdown()
    
if __name__ == '__main__':
    main()