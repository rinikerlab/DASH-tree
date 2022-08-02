Welcome to Serenityff-Charge
==============================

[//]: # (Badges)
[![CI](https://github.com/MTLehner/serenityff-charge/actions/workflows/CI.yaml/badge.svg)](https://github.com/MTLehner/serenityff-charge/actions/workflows/CI.yaml)
[![pre-commit](https://github.com/MTLehner/serenityff-charge/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/MTLehner/serenityff-charge/actions/workflows/pre-commit.yml)
[![codecov](https://codecov.io/gh/MTLehner/serenityff-charge/branch/main/graph/badge.svg?token=8pBnMMfYIg)](https://codecov.io/gh/MTLehner/serenityff-charge)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/MTLehner/serenityff-charge.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/MTLehner/serenityff-charge/context:python)


Description
-------------

   Welcome to Serenityff-Charge. This repository is a collection of scripts, tools and other resources for partial charges in MD simulations.

   It contains tools to generate partial charges for given molecules quickly, using the serenityff-charge workflow and using openff to generate all other parameters. A pre computed decision tree can be used to generate charges for a given molecule.

   Additionally, this repository contains all tools and functions needed to generate a new decision tree for partial charge assignmend, based on the attention data of a graph neural network, cabable of predicting the partial charges of a molecule.

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

* **Decision Tree**
    * Generate decision tree (attention based)
    * Tools for decision tree
        * file I/O
        * pruning
        * statistics
        * assigning charges
    * Normalization of charges

* **OpenFF plugin**
    * Add tree to OpenFF charge assignment

* **OpenFF-Evaluator**
    * Tools for validation of the charges with OpenFF-Evaluator
