#!/usr/bin/env python3

import numpy as np
import anndata as an
import refined_tangram as tg
import sys
import squidpy as sq

adata_st = sq.datasets.visium_fluo_adata_crop()
adata_st = adata_st[
    adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
].copy()

modification = sys.argv[1]

if "cytospace_pp" in modification:
    adata_sc = an.read_h5ad("../data/visium_mouse_cortex_sc_cytospace_pp.h5ad")
elif "constrained_pp" in modification:
    adata_sc = an.read_h5ad("../data/visium_mouse_cortex_sc_constrained_pp.h5ad")
else:
    adata_sc = sq.datasets.sc_mouse_cortex()

genes = None 

if modification == "hvg":
    with open('../data/visium_mouse_cortex_hvg.txt') as f:
        genes = eval(f.readline())
elif modification == "svg":
    with open('../data/visium_mouse_cortex_svg.txt') as f:
        genes = eval(f.readline())
elif modification == "ctg":
    with open('../data/visium_mouse_cortex_ctg.txt') as f:
        genes = eval(f.readline())
elif modification == "hvg+svg+ctg":
    with open('../data/visium_mouse_cortex_hvg+svg+ctg.txt') as f:
        genes = eval(f.readline())
elif modification == "spapros" or "combi" in modification:
    with open('../data/visium_mouse_cortex_spapros.txt') as f:
        genes = eval(f.readline())
else:
        genes = None

tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True, genes=genes)

lambda_g1 = 1.0
lambda_sparsity_g1 = 0
lambda_neighborhood_g1 = 0
lambda_ct_islands = 0
lambda_getis_ord = 0
lambda_moran = 0
lambda_geary = 0
lambda_r = 0
lambda_l1 = 0
lambda_l2 = 0
learning_rate = 0.1

if modification == "sparsity_cosine_similarity":
    lambda_sparsity_g1 = 1.0
if modification == "neighborhood_cosine_similarity":
    lambda_neighborhood_g1 = 1.0
if modification == "ct_island":
    lambda_ct_islands = 1.0
if modification == "moran_I":
    lambda_moran = 1.0
if modification == "geary_C":
    lambda_geary = 1.0
if modification == "getis_ord_g_star":
    lambda_getis_ord = 1.0
if modification == "l1_reg":
    lambda_l1 = 1.4e-07
if modification == "l2_reg":
    lambda_l2 = 1.4e-07
if modification == "entropy_reg":
    lambda_r = 1.4e-07
if "combi" in modification:
    learning_rate = 0.72
    lambda_g1 = 0.99
    lambda_g2 = 0.99
    lambda_neighborhood_g1 = 0.96
    lambda_ct_islands = 0.17
    lambda_getis_ord = 0.71
    lambda_r = 2.95e-09
    lambda_l2 = 1.00e-18

adata_maps_pred = dict()
for run in range(10):
    adata_maps_pred[run] = tg.map_cells_to_space(
        adata_sc, adata_st,
        mode="cells",
        density_prior='rna_count_based',
        num_epochs = 1000,
        lambda_g1 = lambda_g1,
        lambda_neighborhood_g1 = lambda_neighborhood_g1,
        lambda_ct_islands = lambda_ct_islands,
        lambda_getis_ord = lambda_getis_ord,
        lambda_moran = lambda_moran,
        lambda_geary = lambda_geary,
        lambda_sparsity_g1 = lambda_sparsity_g1,
        lambda_r = lambda_r,
        lambda_l1 = lambda_l1,
        lambda_l2 = lambda_l2,
        learning_rate = learning_rate,
        device = 'cuda:0',
        random_state = run
    )
    np.save(f"../models/visium_mouse_cortex/{modification}/{run}.npy", adata_maps_pred[run].X)