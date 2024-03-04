# typing tools for the use of rdkit
from rdkit import Chem
from typing import TypeAlias

Molecule: TypeAlias = Chem.rdchem.Mol
Atom: TypeAlias = Chem.rdchem.Atom
Bond: TypeAlias = Chem.rdchem.Bond
