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

adata_sc = an.read_h5ad("data/merfish_mouse_hypothalamus_sc.h5ad")
adata_st = an.read_h5ad("data/merfish_mouse_hypothalamus_st.h5ad")

tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True)

adata_maps_pred = dict()
for model in ["orig","random",
              "sparsity_cosine_similarity","neighborhood_cosine_similarity","ct_island",
              "getis_ord_g_star","geary_C","moran_I",
              "l1_reg","l2_reg","entropy_reg",
              "combi",
              ]:
    adata_maps_pred[model] = dict()
    for run in range(10):
        adata_maps_pred[model][run] = an.AnnData(X=np.load(f'final_merfish_mouse_hypothalamus_modifications/{model}/{run}.npy', allow_pickle=True), 
                                                obs=adata_sc.obs,
                                                var=adata_st.obs)
tgx.preprocess(adata_maps_pred, adata_sc, adata_st, cell_types_key="cell_types")
sq.gr.spatial_neighbors(adata_st, set_diag=False)
spatial_similarity = np.array(tgx.spatial_weights(adata_st, diag_zero = False))

test_genes = np.load("data/merfish_mouse_hypothalamus_test_genes.npy", allow_pickle=True).tolist()
train_genes = np.load("data/merfish_mouse_hypothalamus_train_genes.npy", allow_pickle=True).tolist()

adata_sc.uns["test_genes"] = adata_sc.uns["overlap_genes"]
adata_maps_true = an.read_h5ad("data/merfish_mouse_hypothalamus_true_mapping.h5ad")

metrics = tgx.benchmark_metrics(adata_maps_pred, adata_sc, adata_st, spatial_similarity, adata_maps_true=adata_maps_true)
with open('benchmarking/final_merfish_mouse_hypothalamus_modifiations.pkl', 'wb') as f:
    pickle.dump(metrics, f)