import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy

def generate_adata_st(adata_sc, xgrid, ygrid, cell_cover=0.8, min_cell_count=3):
    xbin = np.diff(xgrid[0])[0]
    ybin = np.diff(ygrid[:,0])[0]

    X = np.zeros([xgrid.shape[0] * xgrid.shape[1],adata_sc.shape[1]])
    cell_count = np.full(xgrid.shape[0] * xgrid.shape[1], 0)
    sc_selected_for_adata_st = np.full(adata_sc.shape[0], False)
    sc_selected_for_adata_st_filtered = np.full(adata_sc.shape[0], False)
    sc_grid = np.full(adata_sc.shape[0], None)
    sc_spatial = adata_sc.obsm["spatial"]
    sc_expr = adata_sc.X
    #sc_expr = adata_sc.raw.X

    for i in tqdm(range(xgrid.shape[0])):
        for j in range(xgrid.shape[1]):
            xbmin = xgrid[i][j]-xbin*cell_cover/2
            xbmax = xgrid[i][j]+xbin*cell_cover/2
            ybmin = ygrid[i][j]-ybin*cell_cover/2
            ybmax = ygrid[i][j]+ybin*cell_cover/2
            cell_idxs = []
            for cell_idx in range(adata_sc.shape[0]):
                if sc_spatial[cell_idx,0]>xbmin and sc_spatial[cell_idx,0]<=xbmax and \
                   sc_spatial[cell_idx,1]>ybmin and sc_spatial[cell_idx,1]<=ybmax:
                    X[i*xgrid.shape[1]+j] += sc_expr[cell_idx]
                    cell_count[i*xgrid.shape[1]+j] +=1
                    sc_selected_for_adata_st[cell_idx] = True
                    cell_idxs.append(cell_idx)
            if len(cell_idxs) >= min_cell_count:
                for cell_idx in cell_idxs:
                    sc_selected_for_adata_st_filtered[cell_idx] = True
                    sc_grid[cell_idx] = i*xgrid.shape[1]+j
    
    adata_sc.obs["selected_for_adata_st"] = sc_selected_for_adata_st
    adata_sc.obs["selected_for_adata_st_filtered"] = sc_selected_for_adata_st_filtered
    adata_sc.obs["grid"] = sc_grid
    print(f"proportion of selected cells: {adata_sc.obs['selected_for_adata_st_filtered'].sum() / adata_sc.n_obs}")

    grid_idx = [i*xgrid.shape[1]+j for i in range(xgrid.shape[0]) for j in range(xgrid.shape[1])]
    adata_st = AnnData(X=X, 
                       var=adata_sc.var,
                       obs=pd.DataFrame({"cell_count" : cell_count}, index=grid_idx),
                       obsm={"spatial" : np.append(xgrid.reshape(-1,1), ygrid.reshape(-1,1),axis=1)})
    return adata_st

def cells2spots(adata_sc, adata_st):
    dist = scipy.spatial.distance.cdist(adata_sc.obsm["spatial"],np.array(adata_st.obsm["spatial"]))
    adata_sc.obs["grid"] = adata_st.obs_names[dist.argmin(axis=1)]

def cellsubset2spots(adata_sc, adata_st):
    dist = scipy.spatial.distance.cdist(adata_sc[~adata_sc.obs["selected_for_adata_st_filtered"]].obsm["spatial"],np.array(adata_st.obsm["spatial"]))
    adata_sc[~adata_sc.obs["selected_for_adata_st_filtered"]].obs["grid"] = adata_st.obs_names[dist.argmin(axis=1)]

def true_mapping(adata_sc,adata_st):
    x = pd.DataFrame({"cl": adata_sc.obs["grid"]})
    for i in np.unique(adata_sc.obs["grid"]):
        x[i] = list(map(int, x["cl"] == i))
    del x["cl"]
    true_adata_map = AnnData(X=x[adata_st.obs_names], 
                            obs=adata_sc.obs,
                            var=adata_st.obs)
    return true_adata_map