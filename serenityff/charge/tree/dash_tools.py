from typing import Tuple
import numpy as np
from numba import njit

from rdkit import Chem

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.utils.rdkit_typing import Molecule


#@njit(cache=True, fastmath=True)
def new_neighbors(neighbor_dict, connected_atoms) -> tuple:
    connected_atoms_set = set(connected_atoms)
    new_neighbors_afs = []
    new_neighbors = []
    for rel_atom_idx, atom_idx in enumerate(connected_atoms):
        for neighbor in neighbor_dict[atom_idx]:
            if neighbor[0] not in connected_atoms_set and neighbor[0] not in new_neighbors:
                new_neighbors.append(neighbor[0])
                new_neighbors_afs.append(
                    (neighbor[1][0], rel_atom_idx, neighbor[1][2])
                )  # fix rel_atom_idx (the atom index in the subgraph)
    return new_neighbors_afs, new_neighbors

#@njit(cache=True, fastmath=True)
def new_neighbors_atomic(neighbor_dict, connected_atoms, atom_idx_added) -> tuple:
    connected_atoms_set = set(connected_atoms)
    new_neighbors_afs = []
    new_neighbors = []
    for neighbor in neighbor_dict[atom_idx_added]:
        if neighbor[0] not in connected_atoms_set:
            new_neighbors.append(neighbor[0])
            new_neighbors_afs.append(
                (neighbor[1][0], atom_idx_added, neighbor[1][2])
            )  # fix rel_atom_idx (the atom index in the subgraph)
    return new_neighbors_afs, new_neighbors


def init_neighbor_dict(mol: Molecule):
    neighbor_dict = {}
    for atom_idx, atom in enumerate(mol.GetAtoms()):
        neighbor_dict[atom_idx] = []
        for neighbor in atom.GetNeighbors():
            af_with_connection_info = AtomFeatures.atom_features_from_molecule_w_connection_info(
                mol, neighbor.GetIdx(), (0, atom_idx)
            )
            neighbor_dict[atom_idx].append((neighbor.GetIdx(), af_with_connection_info))
    return neighbor_dict


def get_rdkit_fragment_from_node_path(node_path) -> Chem.RWMol:
    """
    Get the rdkit fragment from a node path as rdkit molecule.

    Args:
        node_path (list[node]): A list of nodes, forming the subgraph/path in the tree.

    Returns:
        Chem.RWMol: The rdkit molecule of the subgraph/path.
    """
    print("WARNING: get_rdkit_fragment_from_node_path is deprecated.")
    # start with an empty molecule
    mol = Chem.RWMol()
    # add the first atom
    element, numBonds, charge, isConjugated, numHs = AtomFeatures.get_split_feature_typed_from_key(
        node_path[0].atoms[0][0]
    )
    atom = Chem.Atom(element)
    atom.SetFormalCharge(charge)
    atom.SetNumExplicitHs(numHs)
    mol.AddAtom(atom)
    # add the rest of the atoms
    for i in range(1, len(node_path)):
        element, numBonds, charge, isConjugated, numHs = AtomFeatures.get_split_feature_typed_from_key(
            node_path[i].atoms[0][0]
        )
        atom = Chem.Atom(element)
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(numHs)
        mol.AddAtom(atom)
        # add the bond
        bonded_atom = node_path[i].atoms[0][1]
        bond_type_number = node_path[i].atoms[0][2]
        mol.AddBond(i, bonded_atom, Chem.rdchem.BondType.values[bond_type_number])
    return mol