from rdkit import Chem


class atom_features:
    """
    class to represent a atom type, as stored in the decision tree

    atom_features are hashed for fast comparison in the decision tree
    """

    def __init__(self, mol: Chem.rdchem.Mol = None, idx=None, connectedTo=(None, None), data=None):
        if data is None and mol is not None and idx is not None:
            atom = mol.GetAtomWithIdx(int(idx))
            self.idx = idx
            self.element = atom.GetSymbol()
            self.num_bonds = len(atom.GetBonds())
            self.formalCharge = int(atom.GetFormalCharge())
            self.hybridization = str(atom.GetHybridization())
            self.IsAromatic = bool(atom.GetIsAromatic())
            self.TotalNumHs = int(atom.GetTotalNumHs())
            self.connectedTo = connectedTo
            self.connectionBondType = (
                None
                if connectedTo[1] is None
                else str(mol.GetBondBetweenAtoms(int(idx), int(connectedTo[1])).GetBondType())
            )
            self.hash = self._hash_atom_features()
        elif data is not None:
            if len(data) == 8:
                self.idx = 0
                self.element = data[0]
                self.num_bonds = int(data[1])
                self.formalCharge = int(data[2])
                # format hybritisation steps
                hybr_str = data[3]
                if hybr_str in Chem.HybridizationType.names:
                    self.hybridization = hybr_str
                else:
                    hybr_int = int(hybr_str[hybr_str.find("(") + 1 : hybr_str.find(")")])
                    self.hybridization = str(Chem.rdchem.HybridizationType.values[hybr_int])
                self.IsAromatic = True if data[4].lower() in ("true", "1") else False
                self.TotalNumHs = int(data[5])
                try:
                    self.connectedTo = (int(data[6]), 0)
                    bondType_str = data[7]
                    if bondType_str in Chem.BondType.names:
                        self.connectionBondType = bondType_str
                    else:
                        bondType_int = int(bondType_str[bondType_str.find("(") + 1 : bondType_str.find(")")])
                        self.connectionBondType = str(Chem.rdchem.BondType.values[bondType_int])
                except ValueError:
                    self.connectedTo = (None, None)
                    self.connectionBondType = None
                self.hash = self._hash_atom_features()
            else:
                raise ValueError("data must be a list of length 8")
        else:
            self.idx = 0
            self.element = "XXX"
            self.num_bonds = 0
            self.formalCharge = 0
            self.hybridization = ""
            self.IsAromatic = False
            self.TotalNumHs = 0
            self.connectedTo = (None, None)
            self.connectionBondType = None
            self.hash = 0

    def __repr__(self):
        return f"{self.element} {self.num_bonds} {self.formalCharge} {str(self.hybridization)} {self.IsAromatic} {self.TotalNumHs} {self.connectedTo[0]} {str(self.connectionBondType)}"

    def __eq__(self, other):
        if self.hash == other.hash:
            return (
                self.element == other.element
                and self.num_bonds == other.num_bonds
                and self.formalCharge == other.formalCharge
                and self.hybridization == other.hybridization
                and self.IsAromatic == other.IsAromatic
                and self.TotalNumHs == other.TotalNumHs
                and self.connectedTo[0] == other.connectedTo[0]
                and self.connectionBondType == other.connectionBondType
            )
        return False

    def __hash__(self):
        return self.hash

    def _update_hash(self):
        self.hash = hash(repr(self))

    def _hash_atom_features(self):
        hash_str = f"{self.element}{self.num_bonds}{self.formalCharge}{str(self.hybridization)}{self.IsAromatic}{self.TotalNumHs}{self.connectedTo[0]}{self.connectionBondType}"
        return hash(hash_str)
