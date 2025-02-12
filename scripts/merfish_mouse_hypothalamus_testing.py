#!/usr/bin/env python3

import numpy as np
import anndata as an
import refined_tangram as tg
import benchmarking
import pickle

adata_sc = an.read_h5ad("../data/merfish_mouse_hypothalamus_sc.h5ad")
adata_st = an.read_h5ad("../data/merfish_mouse_hypothalamus_st.h5ad")

tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True)

adata_maps_pred = dict()
for model in ["orig","random",
              "sparsity_cosine_similarity","neighborhood_cosine_similarity","ct_island",
              "getis_ord_g_star","geary_C","moran_I",
              "l1_reg","l2_reg","entropy_reg",
              "combi"]:
    adata_maps_pred[model] = dict()
    for run in range(10):
        adata_maps_pred[model][run] = an.AnnData(X=np.load(f'../models/merfish_mouse_hypothalamus/{model}/{run}.npy', allow_pickle=True), 
                                                obs=adata_sc.obs,
                                                var=adata_st.obs)
        tg.cell_type_mapping(adata_maps_pred[model][run])

adata_sc.uns["test_genes"] = adata_sc.uns["overlap_genes"]
adata_maps_true = an.read_h5ad("../data/merfish_mouse_hypothalamus_true_mapping.h5ad")
metrics = benchmarking.eval_metrics(adata_maps_pred, adata_sc, adata_st, adata_maps_true=adata_maps_true)

with open('../metrics/merfish_mouse_hypothalamus.pkl', 'wb') as f:
    pickle.dump(metrics, f)