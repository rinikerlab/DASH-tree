# Copyright (C) 2022-2025 ETH Zurich, Niels Maeder and other DASH contributors.

"""Typing tools for the use of rdkit."""

from rdkit import Chem

Molecule = Chem.rdchem.Mol
Atom = Chem.rdchem.Atom
Bond = Chem.rdchem.Bond
