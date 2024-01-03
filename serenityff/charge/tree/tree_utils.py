from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit

from rdkit import Chem

from serenityff.charge.tree.atom_features import AtomFeatures
#from serenityff.charge.tree.node import node
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


# def create_new_node_from_develop_node(current_develop_node: DevelopNode) -> node:
#     """
#     Create a new node from a develop node. Used to convert a the raw construction tree to a final tree.

#     Parameters
#     ----------
#     current_develop_node : DevelopNode
#         The develop node to convert.
#     current_new_node : node
#         The new node to create.

#     Returns
#     -------
#     node
#         The new node.
#     """
#     # get current node properties
#     atom = current_develop_node.atom_features
#     level = current_develop_node.level
#     (
#         result,
#         std,
#         attention,
#         size,
#     ) = current_develop_node.get_node_result_and_std_and_attention_and_length()
#     current_new_node = node(
#         atom=[atom],
#         level=level,
#         result=result,
#         stdDeviation=std,
#         attention=attention,
#         count=size,
#     )
#     # do the same things recursively for the children (get their properties and create the new nodes)
#     for child in current_develop_node.children:
#         current_new_node.add_child(create_new_node_from_develop_node(child))
#     return current_new_node


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
    atom_type, con_atom, con_type = atom
    tree_storage.append((id_counter, atom_type, con_atom, con_type, mean_attention, []))
    data_storage.append((level, atom_type, con_atom, con_type, result, std, max_attention, size))
    parent_id = id_counter
    for child in dev_node.children:
        id_counter += 1
        tree_storage[parent_id][5].append(id_counter)
        id_counter = recursive_DEV_node_to_DASH_tree(child, id_counter, parent_id, tree_storage, data_storage)
    return id_counter


def get_DASH_tree_from_DEV_tree(dev_root: DevelopNode, tree_folder_path: str = "./"):
    tree_storage = {}
    data_storage = []
    for child_id, child in enumerate(dev_root.children):
        branch_tree_storage = []
        branch_data_storage = []
        recursive_DEV_node_to_DASH_tree(child, 0, 0, branch_tree_storage, branch_data_storage)
        branch_data_df = pd.DataFrame(branch_data_storage, columns=["level", "atom_type", "con_atom", "con_type", "result", "stdDeviation", "max_attention", "size"])
        tree_storage[child_id] = branch_tree_storage
        data_storage.append(branch_data_df)
    tree = DASHTree(tree_folder_path=tree_folder_path, preload=False)
    tree.data_storage = data_storage
    tree.tree_storage = tree_storage
    #print("tree_storage: ", tree_storage)
    #print("data_storage: ", data_storage)
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