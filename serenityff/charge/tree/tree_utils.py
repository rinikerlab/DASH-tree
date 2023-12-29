from typing import Tuple

import numpy as np
from numba import njit

from rdkit import Chem

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.node import node
from serenityff.charge.tree_develop.develop_node import DevelopNode
from serenityff.charge.utils.rdkit_typing import Molecule
from serenityff.charge.tree.dash_tree import DASHTree


def get_possible_atom_features(mol, connected_atoms):
    """
    Get the possible atom features for a molecule in reach of the connected atoms.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the possible atom features for.
    connected_atoms : list
        The already connected atoms of the subgraph of the molecule.

    Returns
    -------
    list[list[AtomFeatures], list[int]]
        The possible atom features for the molecule and the possible atom idxs.
    """
    possible_atom_features = []
    possible_atom_idxs = []
    for atom in connected_atoms:
        for bond in mol.GetAtomWithIdx(atom).GetBonds():
            if bond.GetBeginAtomIdx() not in connected_atoms:
                possible_atom_features.append(
                    AtomFeatures.atom_features_from_molecule_w_connection_info(
                        mol,
                        bond.GetBeginAtomIdx(),
                        connected_to=(connected_atoms.index(atom), atom),
                    )
                )
                possible_atom_idxs.append(bond.GetBeginAtomIdx())
            if bond.GetEndAtomIdx() not in connected_atoms:
                possible_atom_features.append(
                    AtomFeatures.atom_features_from_molecule_w_connection_info(
                        mol,
                        bond.GetEndAtomIdx(),
                        connected_to=(connected_atoms.index(atom), atom),
                    )
                )
                possible_atom_idxs.append(bond.GetEndAtomIdx())
    return possible_atom_features, possible_atom_idxs


def create_new_node_from_develop_node(current_develop_node: DevelopNode) -> node:
    """
    Create a new node from a develop node. Used to convert a the raw construction tree to a final tree.

    Parameters
    ----------
    current_develop_node : DevelopNode
        The develop node to convert.
    current_new_node : node
        The new node to create.

    Returns
    -------
    node
        The new node.
    """
    # get current node properties
    atom = current_develop_node.atom_features
    level = current_develop_node.level
    (
        result,
        std,
        attention,
        size,
    ) = current_develop_node.get_node_result_and_std_and_attention_and_length()
    current_new_node = node(
        atom=[atom],
        level=level,
        result=result,
        stdDeviation=std,
        attention=attention,
        count=size,
    )
    # do the same things recursively for the children (get their properties and create the new nodes)
    for child in current_develop_node.children:
        current_new_node.add_child(create_new_node_from_develop_node(child))
    return current_new_node


def get_data_from_DEV_node(dev_node: DevelopNode):
    #dev_node.update_average()
    atom = dev_node.atom_features
    level = dev_node.level
    (
        result,
        std,
        max_attention,
        mean_attention,
        size,
    ) = dev_node.get_DASH_data_from_dev_node()
    return (atom, level, result, std, max_attention, mean_attention, size)

def recursive_DEV_node_to_DASH_tree(dev_node: DevelopNode, id_counter: int, parent_id: int, tree_storage: list, data_storage: list):
    # check if tree_storage length is equal to id_counter
    if len(tree_storage) != id_counter:
        print("ERROR: tree_storage length is not equal to id_counter")
        return
    atom, level, result, std, max_attention, mean_attention, size = get_data_from_DEV_node(dev_node)
    atom_type, con_atom, con_type = atom[0]
    tree_storage.append([id_counter, atom_type, con_atom, con_type, mean_attention, [], parent_id])
    data_storage.append((level, atom_type, con_atom, con_type, result, std, max_attention, size))
    for child in dev_node.children:
        id_counter += 1
        tree_storage[id_counter][5].append(id_counter)
        id_counter = recursive_DEV_node_to_DASH_tree(child, id_counter, id_counter-1, tree_storage, data_storage)
    return id_counter


def get_DASH_tree_from_DEV_tree(dev_root: DevelopNode):
    tree_storage = []
    data_storage = []
    for child in dev_root.children:
        recursive_DEV_node_to_DASH_tree(child, 0, 0, tree_storage, data_storage)
    tree = DASHTree(tree_folder_path="./", preload=False)
    tree.data_storage = data_storage
    tree.tree_storage = tree_storage
    tree.save_all_trees_and_data()
        

@njit
def get_possible_connected_new_atom(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Get indices of every neighbor of every atom in indices that is not in indices.

    Args:
        matrix (np.ndarray): connectivity matrix of molecule.
        indices (np.ndarray): already connected atoms.

    Returns:
        np.ndarray: indices of new neighbors
    """
    begin = np.zeros(matrix.shape[0], dtype=np.bool_)
    final = np.zeros(matrix.shape[0], dtype=np.bool_)
    for index in indices:
        final += matrix[index]
        begin[index] = True
    return np.where(final ^ begin)[0]


@njit
def get_connected_atom_with_max_attention(
    matrix: np.ndarray, attentions: np.ndarray, indices: np.ndarray
) -> Tuple[float, int]:
    """
    returns idx and attention of atom with highest attention that is connected and not in indices.
    if condition cannot be met, (None, 0) is returned.

    Args:
        matrix (np.ndarray): connectivity matrix of molecule
        attentions (np.ndarray): _description_
        indices (np.ndarray): _description_

    Returns:
        Tuple[float, int]: max attention, idx of atom with max attention
    """
    possible_atoms = get_possible_connected_new_atom(matrix=matrix, indices=indices)
    possible_attentions = np.take(attentions, possible_atoms)
    try:
        return possible_atoms[np.argmax(possible_attentions)], np.max(possible_attentions)
    except Exception:
        return (np.NaN, np.NaN)


@njit
def get_connected_neighbor(matrix: np.ndarray, idx: int, indices: np.ndarray):
    """
    Get the relative and (rdkit) absolute index of the smallest neighbor of idx that is in the subgraph (indices)

    Args:
        matrix (np.ndarray): connectivity matrix of molecule
        idx (int): index of the atom
        indices (np.ndarray): list of atom indices of the subgraph

    Returns:
        (int, int): relative index of the neighbor, absolute index of the neighbor
    """
    neighbors = np.where(matrix[idx])[0]
    for rel_index, absolute_index in enumerate(indices):
        where = np.where(absolute_index == neighbors)[0]
        if where.size > 0:
            return rel_index, absolute_index


def get_rdkit_fragment_from_node_path(node_path) -> Chem.RWMol:
    """
    Get the rdkit fragment from a node path as rdkit molecule.

    Args:
        node_path (list[node]): A list of nodes, forming the subgraph/path in the tree.

    Returns:
        Chem.RWMol: The rdkit molecule of the subgraph/path.
    """
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
