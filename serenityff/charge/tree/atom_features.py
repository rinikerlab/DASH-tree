from typing import Any, Optional, Sequence, Tuple

from rdkit import Chem

from serenityff.charge.utils import Molecule


class AtomFeatures:
    """
    class to represent an atom type, as stored in the decision tree
    AtomFeatures are hashed for fast comparison in the decision tree
    """

    def __init__(
        self,
        idx: int = 0,
        element: str = "NAN",
        num_bonds: int = 0,
        formal_charge: int = 0,
        hybridization: str = "",
        is_aromatic: bool = False,
        total_num_hs: int = 0,
        connected_to: Tuple[int] = (None, None),
        conenection_bond_type: str = None,
    ) -> None:
        self.idx = idx
        self.element = element
        self.num_bonds = num_bonds
        self.formal_charge = formal_charge
        self.hybridization = hybridization
        self.is_aromatic = is_aromatic
        self.total_num_hs = total_num_hs
        self.connected_to = connected_to
        self.conenection_bond_type = conenection_bond_type
        self.hash = self._hash()
        return

    @classmethod
    def from_data(cls, data: Sequence[Any]):
        """
        Generates AtomFeatures object from data array. Highly specific. Use from_molecule in most cases.
        Args:
            data (Sequence[Any]): Data to be transformed into AtomFeatures object.
        Raises:
            ValueError: thrown if array length != 8.
        Returns:
            AtomFeatures: AtomFeatures object.
        """
        if not len(data) == 8:
            raise ValueError("Data has to have a length of 8.")
        try:
            connected_to = (int(data[6]), 0)
        except (TypeError, ValueError):
            connected_to = (None, None)
        try:
            bond_type_str = data[7]
            conenection_bond_type = (
                bond_type_str
                if bond_type_str in Chem.BondType.names
                else str(
                    Chem.rdchem.BondType.values[
                        int(bond_type_str[bond_type_str.find("(") + 1 : bond_type_str.find(")")])
                    ]
                )
            )
        except (AttributeError, ValueError):
            conenection_bond_type = None

        return cls(
            idx=0,
            element=data[0],
            num_bonds=data[1],
            formal_charge=data[2],
            hybridization=data[3],
            is_aromatic=True if data[4].lower in ("true", "yes", "1") else False,
            total_num_hs=data[5],
            connected_to=connected_to,
            conenection_bond_type=conenection_bond_type,
        )

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        idx: int,
        connected_to: Optional[Tuple[Any]] = (None, None),
    ):
        """
        Turns atom with idx from molecule into AtomFeatures object.
        Args:
            molecule (Molecule): rdkit Molecule
            idx (int): idx of atom in molecule.
            connected_to (Optional[Tuple[Any]], optional): Neighbouring atoms. Defaults to (None, None).
        Returns:
            AtomFeatures: AtomFeatures object.
        """
        atom = molecule.GetAtomWithIdx(int(idx))
        connected_bond_type = (
            None
            if connected_to[1] is None
            else str(molecule.GetBondBetweenAtoms(int(idx), int(connected_to[1])).GetBondType())
        )
        return cls(
            idx=idx,
            element=atom.GetSymbol(),
            num_bonds=len(atom.GetBonds()),
            formal_charge=int(atom.GetFormalCharge()),
            hybridization=str(atom.GetHybridization()),
            is_aromatic=bool(atom.GetIsAromatic()),
            total_num_hs=int(atom.GetTotalNumHs()),
            connected_to=connected_to,
            conenection_bond_type=connected_bond_type,
        )

    def __repr__(self) -> str:
        return f"{self.element} {self.num_bonds} {self.formal_charge} {str(self.hybridization)} {self.is_aromatic} {self.total_num_hs} {self.connected_to[0]} {str(self.conenection_bond_type)}"

    def __eq__(self, other: object) -> bool:
        if self.hash == other.hash:
            return (
                self.element == other.element
                and self.num_bonds == other.num_bonds
                and self.formal_charge == other.formal_charge
                and self.hybridization == other.hybridization
                and self.is_aromatic == other.is_aromatic
                and self.total_num_hs == other.total_num_hs
                and self.connected_to[0] == other.connected_to[0]
                and self.conenection_bond_type == other.conenection_bond_type
            )
        else:
            return False

    def __hash__(self) -> int:
        return self.hash

    def _hash(self) -> int:
        return hash(repr(self))

    def _is_similar(self, other):
        sum_similar = 0
        if self.element == other.element:
            sum_similar += 1
        if self.num_bonds == other.num_bonds:
            sum_similar += 1
        if self.formal_charge == other.formal_charge:
            sum_similar += 1
        if self.hybridization == other.hybridization:
            sum_similar += 1
        if self.is_aromatic == other.is_aromatic:
            sum_similar += 1
        if self.total_num_hs == other.total_num_hs:
            sum_similar += 1
        return sum_similar / 6

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def element(self) -> str:
        return self._element

    @property
    def num_bonds(self) -> int:
        return self._num_bonds

    @property
    def formal_charge(self) -> int:
        return self._formal_charge

    @property
    def hybridization(self) -> str:
        return self._hybridization

    @property
    def is_aromatic(self) -> bool:
        return self._is_aromatic

    @property
    def total_num_hs(self) -> int:
        return self._total_num_hs

    @property
    def connected_to(self) -> Tuple[Any]:
        return self._connected_to

    @property
    def conenection_bond_type(self) -> str:
        return self._conenection_bond_type

    @property
    def hash(self) -> int:
        return self._hash

    @idx.setter
    def idx(self, value: int) -> None:
        if isinstance(value, int):
            self._idx = value
        elif isinstance(value, float) and value.is_integer():
            self._idx = int(value)
        elif isinstance(value, str):
            try:
                value = float(value)
                self.idx = value
            except ValueError:
                raise TypeError("Idx has to be of type int")
        else:
            raise TypeError("Idx has to be of type int")
        return

    @element.setter
    def element(self, value: str) -> None:
        if isinstance(value, str):
            self._element = value
            return
        else:
            raise TypeError("element has to be of type str")

    @num_bonds.setter
    def num_bonds(self, value: int) -> None:
        if isinstance(value, int):
            self._num_bonds = value
        elif isinstance(value, float) and value.is_integer():
            self._num_bonds = int(value)
        elif isinstance(value, str):
            try:
                value = float(value)
                self.num_bonds = value
            except ValueError:
                raise TypeError("num_bonds has to be of type int")
        else:
            raise TypeError("num_bonds has to be of type int")
        return

    @formal_charge.setter
    def formal_charge(self, value: int) -> None:
        if isinstance(value, int):
            self._formal_charge = value
        elif isinstance(value, float) and value.is_integer():
            self._formal_charge = int(value)
        elif isinstance(value, str):
            try:
                value = float(value)
                self.formal_charge = value
            except ValueError:
                raise TypeError("formal_charge has to be of type int")
        else:
            raise TypeError("formal_charge has to be of type int")
        return

    @hybridization.setter
    def hybridization(self, value: str) -> None:
        if isinstance(value, str):
            if value in Chem.HybridizationType.names:
                self._hybridization = value
            elif (
                str(Chem.rdchem.HybridizationType.values[int(value[value.find("(") + 1 : value.find(")")])])
                in Chem.HybridizationType.names
            ):
                self._hybridization = value
            else:
                self._hybridization = "OTHER"
        else:
            raise TypeError("hybridization has to be of type str")
        return

    @is_aromatic.setter
    def is_aromatic(self, value: bool) -> None:
        if isinstance(value, bool):
            self._is_aromatic = value
        elif isinstance(value, str):
            if value.lower() in ["true", "1"]:
                self._is_aromatic = True
            else:
                self._is_aromatic = False
        else:
            self._is_aromatic = bool(value)

    @total_num_hs.setter
    def total_num_hs(self, value: int) -> None:
        if isinstance(value, int):
            self._total_num_hs = value
        elif isinstance(value, float) and value.is_integer():
            self._total_num_hs = int(value)
        elif isinstance(value, str):
            try:
                value = float(value)
                self.total_num_hs = value
            except ValueError:
                raise TypeError("total_num_hs has to be of type int")
        else:
            raise TypeError("total_num_hs has to be of type int")
        return

    @connected_to.setter
    def connected_to(self, value: Tuple[Any]) -> None:
        if isinstance(value, tuple):
            self._connected_to = value
        elif value == (None, None):
            self._connected_to = value
        else:
            raise TypeError("connected_to has to be of type Tuple")
        return

    @conenection_bond_type.setter
    def conenection_bond_type(self, value: str) -> None:
        if isinstance(value, str):
            self._conenection_bond_type = value
        elif value is None:
            self._conenection_bond_type = value
        else:
            raise TypeError("conenection_bond_type has to be of type str")
        return

    @hash.setter
    def hash(self, value: int) -> None:
        if isinstance(value, int):
            self._hash = value
        elif isinstance(value, float) and value.is_integer():
            self._hash = int(value)
        elif isinstance(value, str):
            try:
                value = float(value)
                self.hash = value
            except ValueError:
                raise TypeError("hash has to be of type int")
        else:
            raise TypeError("hash has to be of type int")
        return

    def _update_hash(self) -> None:
        """
        update stored hash.
        """
        self.hash = hash(self)
        return
