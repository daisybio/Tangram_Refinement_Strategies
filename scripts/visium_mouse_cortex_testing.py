#!/usr/bin/env python3

import numpy as np
import anndata as an
import refined_tangram as tg
import benchmarking
import pickle
import squidpy as sq

adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[
    adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
].copy()
adata_st.X = adata_st.X.todense()

adata_sc = sq.datasets.sc_mouse_cortex()

tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True)

adata_maps_pred = dict()
for model in ["orig",
              "random",
              "hvg","ctg","svg","hvg+svg+ctg","spapros",
              "sparsity_cosine_similarity",
              "neighborhood_cosine_similarity","ct_island",
              "getis_ord_g_star","geary_C","moran_I",
              "l1_reg","l2_reg","entropy_reg",
              "combi"]:
    adata_maps_pred[model] = dict()
    for run in range(10):
        adata_maps_pred[model][run] = an.AnnData(X=np.load(f'../models/visium_mouse_cortex/{model}/{run}.npy', allow_pickle=True), 
                                                obs=adata_sc.obs,
                                                var=adata_st.obs)

    tg.cell_type_mapping(adata_maps_pred[model][run])

adata_sc.uns["test_genes"] = adata_sc.uns["overlap_genes"]
adata_maps_true = None
metrics = benchmarking.eval_metrics(adata_maps_pred, adata_sc, adata_st, adata_maps_true=adata_maps_true)

with open('../metrics/visium_mouse_cortex.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# constrained pp
adata_sc = an.read_h5ad("../data/visium_mouse_cortex_sc_constrained_pp.h5ad")
tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True)

adata_maps_pred = dict()
for model in ["constrained_pp",
              "constrained_pp_combi"]:
    adata_maps_pred[model] = dict()
    for run in range(10):
        adata_maps_pred[model][run] = an.AnnData(X=np.load(f'../models/visium_mouse_cortex/{model}/{run}.npy', allow_pickle=True), 
                                                obs=adata_sc.obs,
                                                var=adata_st.obs)

    tg.cell_type_mapping(adata_maps_pred[model][run])

adata_sc.uns["test_genes"] = adata_sc.uns["overlap_genes"]
adata_maps_true = None
metrics = benchmarking.eval_metrics(adata_maps_pred, adata_sc, adata_st, adata_maps_true=adata_maps_true)

with open('../metrics/visium_mouse_cortex_constrained_pp.pkl', 'wb') as f:
    pickle.dump(metrics, f)

# cytospace pp
adata_sc = an.read_h5ad("../data/visium_mouse_cortex_sc_cytospace_pp.h5ad")
tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True)

adata_maps_pred = dict()
for model in ["cytospace_pp",
              "cytospace_pp_combi"]:
    adata_maps_pred[model] = dict()
    for run in range(10):
        adata_maps_pred[model][run] = an.AnnData(X=np.load(f'../models/visium_mouse_cortex/{model}/{run}.npy', allow_pickle=True), 
                                                obs=adata_sc.obs,
                                                var=adata_st.obs)

    tg.cell_type_mapping(adata_maps_pred[model][run])

adata_sc.uns["test_genes"] = adata_sc.uns["overlap_genes"]
adata_maps_true = None
metrics = benchmarking.eval_metrics(adata_maps_pred, adata_sc, adata_st, adata_maps_true=adata_maps_true)

with open('../metrics/visium_mouse_cortex_cytospace_pp.pkl', 'wb') as f:
    pickle.dump(metrics, f)