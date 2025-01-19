#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as an
import matplotlib.pyplot as plt
import sklearn
import tangram_modified_version as tg
import tangram_add_ons as tgx
import pickle 
import sys
import os
import squidpy as sq

modification = sys.argv[1]
print(f"use {modification} modification")

adata_sc = an.read_h5ad("data/seqfish_mouse_embryo_sc.h5ad")
adata_st = an.read_h5ad("data/seqfish_mouse_embryo_st.h5ad")

tg.pp_adatas(adata_sc, adata_st, gene_to_lowercase=True)

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

sq.gr.spatial_neighbors(adata_st, set_diag=False)
spatial_weights = tgx.getis_ord_G_star_spatial_weights(adata_st)

if modification == "sparsity_cosine_similarity":
    lambda_sparsity_g1 = 1.0
if modification == "neighborhood_cosine_similarity":
    lambda_neighborhood_g1 = 1.0
if modification == "ct_island":
    lambda_ct_islands = 1.0
if modification == "moran_I":
    lambda_moran = 1.0
    spatial_weights = tgx.spatial_weights(adata_st, diag_zero = True)
if modification == "geary_C":
    lambda_geary = 1.0
    spatial_weights = tgx.spatial_weights(adata_st, diag_zero = True)
if modification == "getis_ord_g_star":
    lambda_getis_ord = 1.0
    spatial_weights = tgx.getis_ord_G_star_spatial_weights(adata_st)
if modification == "l1_reg":
    lambda_l1 = 3.1e-09#1e-7
if modification == "l2_reg":
    lambda_l2 = 3.1e-09
if modification == "entropy_reg":
    lambda_r = 3.1e-09
if modification == "combi":
    learning_rate = 0.72
    lambda_g1 = 0.99
    lambda_g2 = 0.99
    lambda_neighborhood_g1 = 0.96
    lambda_ct_islands = 0.17
    lambda_getis_ord = 0.71
    lambda_r = 2.95e-11#2.95e-09
    lambda_l2 = 1.00e-20#1.00e-18
voxel_weights = tgx.spatial_weights(adata_st, diag_zero = False)
neighborhood_filter = np.array(adata_st.obsp["spatial_connectivities"].todense())
ct_encode = tg.one_hot_encoding(adata_sc.obs["cell_types"]).values

try:
    os.mkdir(f"final_seqfish_mouse_embryo_modifiations/{modification}/") 
except OSError as error:
    print(error)

adata_maps_pred = dict()
for run in range(10):
    adata_maps_pred[run] = tg.map_cells_to_space(
        adata_sc, adata_st,
        mode="cells",
        density_prior='rna_count_based',
        lambda_g1=lambda_g1,
        lambda_neighborhood_g1=lambda_neighborhood_g1,
        voxel_weights=voxel_weights,
        lambda_ct_islands = lambda_ct_islands,
        neighborhood_filter = neighborhood_filter,
        ct_encode = ct_encode,
        lambda_getis_ord=lambda_getis_ord,
        lambda_moran = lambda_moran,
        lambda_geary = lambda_geary,
        spatial_weights=spatial_weights,
        lambda_sparsity_g1=lambda_sparsity_g1,
        num_epochs=1000,
        lambda_r = lambda_r,
        lambda_l1 = lambda_l1,
        lambda_l2 = lambda_l2,
        learning_rate=learning_rate,
        device='cuda:0',
        random_state=run
    )
    np.save(f"final_seqfish_mouse_embryo_modifiations/{modification}/{run}.npy", adata_maps_pred[run].X)