# Refinement Strategies for Tangram

![Abstract](figures/abstract.png)

## Overview

[Tangram](https://github.com/broadinstitute/Tangram/) aligns single-cell and spatial data by comparing gene expression of shared genes via the cosine similarity for single-cell to spatial mapping in its default setting.
The simplicity of the model allows the incorporation of other terms to add, e.g., prior knowledge.

We refined Tangram including (1) optimizing gene set selection, (2) employing regularization techniques to balance consistency and certainty,  (3) incorporating spatial information using, e.g., neighborhood-based indicators, and (4) testing strategies for improved cell subset selection.

Evaluations on real and simulated mouse datasets demonstrated that this approach improves both gene expression prediction and cell(type) mapping. 

![Pipeline](figures/pipeline.png)


## Installation

Set up conda environment using the `environment.yml` file
```
    conda env create -f environment.yml
    conda activate tangramx-env
```

## Usage

To start using Tangram with our refinements, gene selection, or our benchmarking framework, import the code in your jupyter notebooks or/and scripts 
```
    import refined_tangram as tg
    import gene_selection
    import cell_selection
    import benchmarking
```

### Single-Cell to Spatial Mapping

The refinements build on Tangram’s code, keeping its usage unchanged while providing additional options through new hyperparameters and functions.

Load your spatial data and your single cell data, and pre-process them using Tangram's function `tg.pp_adatas`.

To select a specifc gene set, you can use the following functions beforehand:
- `gene_selection.ctg(adata_sc)` for cell type specific genes,
- `gene_selection.hvg(adata_sc)` for highly variable genes of the single-cell dataset,
- `gene_selection.spapros(adata_sc)` for the probe set selected via [Spapros](https://github.com/theislab/spapros), or
- `gene_selection.svg(adata_sp)` for spatially variable genes computed via [SpatialDE2](https://github.com/PMBio/SpatialDE).
```
    adata_sp = sc.read_h5ad(<path>)
    adata_sc = sc.read_h5ad(<path>)

    genes = gene_selection.ctg(adata_sc)
    
    tg.pp_adatas(adata_sc, adata_sp, genes=genes)
```

Once the datasets are pre-processed we can map the single cells onto the spots via Tangram's function `tg.map_cells_to_space`.

Several regularization strategies are available via the hyperparameters `lambda_r`, `lambda_l1`, and `lambda_l2`.

Spatial information can be integrated in the form of spatial weight matrices that capture the locality for each spot. We added three extensions to the loss function based on that:
- Spatially weighted gene expression comparison with the hyperparameter `lambda_neighborhood_g1`,
- Preservation of local spatial indicators with the hyperparameters `lambda_getis_ord` for the local Getis-Ord $G^*$ statistic, `lambda_geary` for the local Geary's $C$ statistic, and `lambda_moran` for the local Moran's $I$ statistic, and
- Enforcement of cell type islands with the hyperparameter `lambda_ct_islands`.

```
    adata_map = tg.map_cells_to_space(adata_sc, adata_sp, lambda_r = 2.95e-09, lambda_l2 = 1e-18, 
                                      lambda_neighborhood_g1 = 0.99, lambda_getis_ord = 0.71, 
                                      lambda_ct_islands = 0.17)
```

The returned `adata_map` is a cell-by-spot structure where `adata_map.X[i, j]` gives the probability for cell ```i``` to be in voxel ```j```. 
These probabilities can be used to derive cell type mapping probabilities with our function `tg.cell_type_mapping(adata_map)`.

Depending on the specific task and dataset, mapping only a subset of the cells may be beneficial.  
`tg.map_cells_to_space` offers the hyperparameter `mode="constrained"` that allows the model to learn an optimal cell subset during training. 
To enable a cell sampling adapted from [CytoSPACE](https://github.com/digitalcytometry/cytospace), we added the function `cell_selection.cell_sampling(adata_sc, adata_sp)` that returns a modified `adata_sc` which can be used with the default `mode="cells"`.

### Dataset Simulation

Since real dataset pairs lack ground truth for cell mapping, we generated low-resolution datasets using data from spatial technologies with single-cell resolution.
This process involves aggregating nearby cells into pseudo-spots based on a spatial grid with the function `benchmarking.generate_adata_st`, followed by assigning each cell to its nearest spot with the function `benchmarking.cells2spots` and finally generating the true mapping object with `benchmarking.true_mapping`.
```
xgrid, ygrid  = np.meshgrid(np.linspace(0, 1, 25),
                            np.linspace(0, 1, 25))
gen_adata_sp = benchmarking.generate_adata_st(adata_sc, xgrid, ygrid, cell_cover=0.8, min_cell_count=3)
benchmarking.cells2spots(adata_sc, gen_adata_sp)
true_adata_map = benchmarking.true_mapping(adata_sc,adata_sp)
```

### Benchmarking

To evaluate correctness, consistency, agreement, and certainty of gene expression prediction, cell, and cell type mapping across multiple runs, you can store the resulting mapping objects in a nested dictionary `adata_maps_pred`. 
Each model should have a unique label as the first key, containing another dictionary where the run number serves as the key.
```
metrics = benchmarking.eval_metrics(adata_maps_pred, adata_sc, adata_st, true_adata_map)
```
The measurements have the same nested dictonary structure.
To get a mean value for each model and metric, you can run `benchmarking.mean_metrics(metrics)`.

### Hyperparameter Tuning

We extended Tangram's framework by enabling hyperparameter tuning using [Ray](https://github.com/ray-project/ray). 
It can be installed via `pip install ray`.

You can use Optuna’s search algorithm to optimize for the correctness, consistency, and / or certainty of the gene expression prediction and / or cell mapping.
```
metric = ["cell_map_consistency","cell_map_agreement","cell_map_certainty",
          "gene_expr_consistency","gene_expr_correctness"]

config = {
    "learning_rate" : tune.loguniform(0.001, 1),
    "lambda_g1": tune.uniform(0, 1.0),
    "lambda_r": tune.loguniform(1e-20, 1e-3),
    "lambda_l2": tune.loguniform(1e-20, 1e-3),
    "lambda_neighborhood_g1": tune.uniform(0, 1.0),
    "lambda_ct_islands": tune.uniform(0, 1.0),
    "lambda_getis_ord": tune.uniform(0, 1.0),
}

tuner = tg.map_cells_to_space_hyperparameter_tuning(adata_sc, adata_sp, metric, config)
tuner.get_results()
```