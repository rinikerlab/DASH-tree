# typing tools for the use of rdkit
from rdkit import Chem
#from typing import TypeAlias
from typing import TypeAlias as TA

Molecule: TA = Chem.rdchem.Mol
Atom: TA = Chem.rdchem.Atom
Bond: TA = Chem.rdchem.Bond
