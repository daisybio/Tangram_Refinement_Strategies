#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as an
import matplotlib.pyplot as plt
import sklearn
import tangram as tg
import tangram_add_ons as tgx
import pickle 
import squidpy as sq

sc.logging.print_header()

adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[
    adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
].copy()
img = sq.datasets.visium_fluo_image_crop()

adata_sc = sq.datasets.sc_mouse_cortex()

tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True)
adata_st.X = adata_st.X.todense()

adata_maps_pred = dict()
for model in ["orig",
              "random",
              "hvg","ctg","svg","hvg+svg+ctg","spapros",
              "sparsity_cosine_similarity",
              "neighborhood_cosine_similarity","ct_island",
              "getis_ord_g_star",
              "geary_C",
              "moran_I",
              "l1_reg","l2_reg","entropy_reg",
              "constrained_pretraining",
              "combi",
              ]:
    adata_maps_pred[model] = dict()
    print(model)
    for run in range(10):
        adata_maps_pred[model][run] = an.AnnData(X=np.load(f'final_visium_mouse_cortex_modifiations_new/{model}/{run}.npy', allow_pickle=True), 
                                                obs=adata_sc.obs,
                                                var=adata_st.obs)

tgx.preprocess(adata_maps_pred, adata_sc, adata_st, cell_types_key="cell_subclass")
sq.gr.spatial_neighbors(adata_st, set_diag=False)
spatial_similarity = np.array(tgx.spatial_weights(adata_st, diag_zero = False))

test_genes = np.load("visium_mouse_cortex_gene_sets_testsplit/test_genes.npy", allow_pickle=True).tolist()
adata_sc.uns["test_genes"] = adata_sc.uns["overlap_genes"]
adata_maps_true = None

metrics = tgx.benchmark_metrics(adata_maps_pred, adata_sc, adata_st, spatial_similarity, adata_maps_true=adata_maps_true)
with open('benchmarking/visium_mouse_cortex_modifications.pkl', 'wb') as f:
    pickle.dump(metrics, f)
