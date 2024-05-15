# typing tools for the use of rdkit
from rdkit import Chem
import typing

Molecule: typing.TypeAlias = Chem.rdchem.Mol
Atom: typing.TypeAlias = Chem.rdchem.Atom
Bond: typing.TypeAlias = Chem.rdchem.Bond
