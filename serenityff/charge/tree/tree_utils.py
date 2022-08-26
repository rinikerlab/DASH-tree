from typing import Tuple

import numpy as np
from numba import njit

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree.node import node
from serenityff.charge.tree_develop.develop_node import DevelopNode


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
                    AtomFeatures.from_molecule(
                        mol,
                        bond.GetBeginAtomIdx(),
                        connected_to=(connected_atoms.index(atom), atom),
                    )
                )
                possible_atom_idxs.append(bond.GetBeginAtomIdx())
            if bond.GetEndAtomIdx() not in connected_atoms:
                possible_atom_features.append(
                    AtomFeatures.from_molecule(
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
    return possible_atoms[np.argmax(possible_attentions)], np.max(possible_attentions)


@njit
def get_connected_neighbor(matrix: np.ndarray, idx: int, indices: np.ndarray):
    """
    #TODO: MARC PLS HELP

    Args:
        matrix (np.ndarray): connectivity matrix of molecule
        idx (int):
        indices (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    neighbors = np.where(matrix[idx])[0]
    for rel_index, absolute_index in enumerate(indices):
        where = np.where(absolute_index == neighbors)[0]
        if where.size > 0:
            return rel_index, absolute_index
