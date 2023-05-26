# DASH Developer Scripts

## Content

1. ```*_allPlots.ipynb``` - Notebook to generate all plots for the paper. For simple examples of the workflow, see the examples folder.
2. ```*_psi4_rest``` - Folder with the scripts to generate all MBIS DFT reference calculations.
3. ```*_train``` - Folder with the scripts to train the GNN. These scripts need pytorch-geometric installed.
4. ```*_explain``` - Folder with the scripts to generate the attention data from the GNN with GNNExplainer. These scripts need pytorch-geometric installed.
5. ```*_tree``` - Folder with the scripts to generate the DASH tree from the attention data and the MBIS reference data. Scripts are RAM intensive (> 64 GB) and optimized to run on a cluster with large memory nodes.
6. ```*_4charges``` - Folder with the scripts to calculate multiple reference charges on the validation set for comparison to the DASH tree.
7. ```*_attThresh``` - Folder with the scripts to generate the DASH tree with different attention thresholds. And compare the RMSE to the reference MBIS charges.
8. ```*_depth``` - Folder with the scripts to generate the DASH tree with different depths. And compare the RMSE to the reference MBIS charges.
