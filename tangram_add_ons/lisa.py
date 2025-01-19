import collections
import sklearn
import libpysal
import numpy as np

def spatial_weights(adata_st, diag_zero = True):
    w = adata_st.obsp['spatial_connectivities'].copy().toarray()
    g = sklearn.preprocessing.normalize(adata_st.obsp['spatial_distances'], norm="l1", axis=1, copy=False).toarray()
    neighbors = collections.defaultdict()
    neighbor_weights = collections.defaultdict()
    for cell in range(w.shape[0]):
        neighbors[cell]=np.where(w[cell]!=0)[0]
        neighbor_weights[cell]=g[cell][np.where(g[cell]!=0)[0]]
        
    w = libpysal.weights.W(neighbors, neighbor_weights)
    spatial_weights = w.sparse.todense()
    if not diag_zero:
        spatial_weights += np.eye(spatial_weights.shape[0])
    return spatial_weights
        
def getis_ord_G_star_spatial_weights(adata_st):
    return adata_st.obsp['spatial_connectivities'].todense() + np.diag(np.ones(adata_st.shape[0]))