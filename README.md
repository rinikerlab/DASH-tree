Welcome to Serenityff-Charge
==============================

[//]: # (Badges)
[![CI](https://github.com/MTLehner/serenityff-charge/actions/workflows/CI.yaml/badge.svg)](https://github.com/MTLehner/serenityff-charge/actions/workflows/CI.yaml)
[![pre-commit](https://github.com/MTLehner/serenityff-charge/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/MTLehner/serenityff-charge/actions/workflows/pre-commit.yml)



Description
-------------

   Welcome to Serenityff-Charge. This repository is a collection of scripts, tools and other resources for partial charges in MD simulations.

   It contains tools to generate partial charges for given molecules quickly, using the serenityff-charge workflow and using openff to generate all other parameters. A pre computed DASH (Dynamic Attention-based Substructure Hierarchy) tree can be used to generate charges for a given molecule.

   Additionally, this repository contains all tools and functions needed to generate a new decision tree for partial charge assignmend, based on the attention data of a graph neural network, cabable of predicting the partial charges of a molecule.

   This repository contains code for the publication by M. Lehner et al. DOI: (TODO: add DOI)


Content
-------------

* **Data Preperation**
    * Select data from Database (Chembel)
    * Generate diverse data set
    * Generate feature vectors

* **Graph neural network**
    * Generate graph neural network (PyTorchGeometric)
    * Train graph neural network
    * Predict partial charges
    * Attention Extraction (GNNExplainer)

* **DASH Tree**
    * Generate DASH tree (attention based)
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
    * Examples for all important functions and tools
    * A good starting point for new users


Installation
-------------

   This repository comes with a conda environment file. To install the environment, run the following command in the root directory of this repository:

   ```bash
    conda env create -f environment.yml
    conda activate serenityff-charge
    python setup.py install
   ```

This will create a conda enviroment with the correct packages and install the openff plugin for partial charge assignment in openff.
