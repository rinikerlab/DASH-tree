Welcome to DASH
==============================

|**Status**| [![CI](https://github.com/rinikerlab/DASH-tree/actions/workflows/CI.yaml/badge.svg)](https://github.com/rinikerlab/DASH-tree/actions/workflows/CI.yaml) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) |
| :------ | :------- |
|**References**| [![DASH](https://img.shields.io/badge/DOI-10.1021/acs.jcim.3c00800-blue)](https://doi.org/10.1021/acs.jcim.3c00800) [![DASH Properties](https://img.shields.io/badge/DOI-10.1063/5.0218154-blue)](https://doi.org/10.1063/5.0218154) |


Description
-------------

   Welcome to DASH. This repository is a collection of scripts, tools and other resources for partial charges in MD simulations.

   It contains tools to generate partial charges and other atomic and molecular properties for given molecules quickly. By using the DASH-charge workflow and using the OpenFF plugin and Forcefield to parametrize molecules quickly with QM-like charge quality. A pre-computed DASH (Dynamic Attention-based Substructure Hierarchy) tree is included and can be used to generate charges for a given molecule. Threes with different properties are available for download from the ETHZ Research Collection.

   Additionally, this repository contains all tools and functions needed to generate a new DASH tree for any property, like partial charge assignments, based on the attention data of a graph neural network, capable of predicting the partial charges of a molecule.

   This repository contains code for the publication by M. Lehner et al. DOI: [arXiv:2305.15981](https://doi.org/10.48550/arXiv.2305.15981), [10.1021/acs.jcim.3c00800](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c00800), and [chemrxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6666e12112188379d8c44aa5).


Content
-------------

* **Data Preperation**
    * Select data from the Database (ChEMBL)
    * Generate diverse data set
    * Generate feature vectors

* **Graph neural network**
    * Generate graph neural network (PyTorchGeometric)
    * Train graph neural network
    * Predict partial charges
    * Attention Extraction (GNNExplainer)

* **DASH Tree**
    * Generate DASH tree (attention-based)
    * Tools for DASH tree
        * file I/O
        * pruning 
        * statistics 
        * assigning charges
    * Normalization of charges

* **OpenFF plugin**
    * Add DASH tree to OpenFF charge assignment

* **OpenFF-Evaluator**
    * Tools for validation of the charges with OpenFF-Evaluator

* **Examples**
    * Examples of all important functions and tools
    * A good starting point for new users


Installation
-------------

   This repository comes with a conda environment file. To install the environment, run the following command in the root directory of this repository:

   ```bash
    conda env create -f min_environment.yml
    conda activate dash
    conda develop .
   ```

This will create a conda environment with the correct packages. The environment contains the minimal dependencies required to use the tree. If the OpenFF plugin to automatically assign charge is required, use the file  `environment.yml` instead. This will create the environment, the DASH package and install the OpenFF plugin for partial charge assignment in OpenFF. The file `environment.yml` also contains all torch dependencies, used for DASH tree development.


Usage
-------------

A default tree for MBIS partial charges is included in the repository. To use the tree, the following code can be used:

```python
# Import the DASH tree and RDKit
from rdkit import Chem
from serenityff.charge.tree.dash_tree import DASHTree
# Load the default tree
tree = DASHTree()
# Create a RDKit molecule
example_mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
# Assign charges
charges = tree.get_molecules_partial_charges(example_mol)["charges"]
```

More atomic and molecular properties can be calculated from a DASH tree populated with these properties.

```python
# Import the DASH tree and RDKit
from rdkit import Chem
from serenityff.charge.tree.dash_tree import DASHTree, TreeType
from serenityff.charge.data import dash_props_tree_path
# Load the property tree.
# Note, that the files will be automatically downloaded the first time the tree is loaded from the ETHZ Research Collection.
tree = DASHTree(tree_folder_path=dash_props_tree_path, tree_type=TreeType.FULL)
# Create a RDKit molecule
example_mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
# Get a new property
tree.get_property_noNAN(mol=example_mol, atom=0, property_name="DFTD4:C6")
# Or get partial charges with a different model
charges = tree.get_molecules_partial_charges(example_mol, chg_key="AM1BCC", chg_std_key="AM1BCC_std")["charges"]
