from rdkit import Chem

from serenityff.charge.tree.atom_features import atom_features
from serenityff.charge.tree.node import node
from serenityff.charge.tree_develop.develop_node import develop_node


def create_mol_from_suply(sdf_suply: str, index: int) -> Chem.Mol:
    """
    Create a molecule from a sdf suply.

    Parameters
    ----------
    sdf_suply : str
        The sdf file path to create the molecule from.
    index : int
        The index of the molecule in the sdf file.

    Returns
    -------
    Chem.Mol
        The molecule with all Hs
    """
    return Chem.SDMolSupplier(sdf_suply, removeHs=False)[index]


def get_possible_connected_atom(atom_idx: int, layer: int, mol: Chem.Mol) -> list:
    """
    Get the possible connected atoms of a subgraph of size layer in the construction tree.

    Parameters
    ----------
    atom_idx : int
        The idx of the atom.
    layer : int
        Number of layers to iterate over.
    mol : Chem.Mol
        The molecule to get the possible connected atoms for.

    Returns
    -------
    _type_
        _description_
    """
    possible_atoms = [int(atom_idx)]
    last_atoms = [int(atom_idx)]
    for _ in range(layer):
        new_atoms = []
        for a in last_atoms:
            for b in mol.GetAtomWithIdx(a).GetBonds():
                if b.GetBeginAtomIdx() not in possible_atoms:
                    possible_atoms.append(b.GetBeginAtomIdx())
                    new_atoms.append(b.GetBeginAtomIdx())
                if b.GetEndAtomIdx() not in possible_atoms:
                    possible_atoms.append(b.GetEndAtomIdx())
                    new_atoms.append(b.GetEndAtomIdx())
        last_atoms = new_atoms
    return possible_atoms


def get_possible_connected_new_atom(mol: Chem.Mol, connected_atoms: list[int]) -> list:
    """
    Get the list of neighbours of a subgraph in the construction tree.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to get the possible connected atoms for.
    connected_atoms : list[int]
        The already connected atoms of the subgraph of the molecule.

    Returns
    -------
    list[int]
        The possible connected atoms of the subgraph of the molecule.
    """
    possible_atoms = connected_atoms
    new_atoms = set()
    for atom in possible_atoms:
        for bond in mol.GetAtomWithIdx(atom).GetBonds():
            if bond.GetBeginAtomIdx() not in possible_atoms:
                new_atoms.add(bond.GetBeginAtomIdx())
            if bond.GetEndAtomIdx() not in possible_atoms:
                new_atoms.add(bond.GetEndAtomIdx())
    return list(new_atoms)


def get_connected_atom_with_max_attention(
    atom_idx: int, layer: int, mol: Chem.Mol, node_attentions: list[float]
) -> list:
    """
    Get the neighbour atom with the highest attention.

    Parameters
    ----------
    atom_idx : int
        The idx of the atom.
    layer : int
        Number of layers to iterate over.
    mol : Chem.Mol
        The molecule to get the possible connected atoms for.
    node_attentions : list[float]
        The attention of the nodes.

    Returns
    -------
    list[int, int]
        The  absolute atom idx of the connected atom with the highest attention and the attention.
    """
    possible_atoms = get_possible_connected_atom(atom_idx, layer, mol)
    max_attention_index = None
    max_attention = 0
    for atom_idx in possible_atoms:
        if node_attentions[atom_idx] > max_attention:
            max_attention_index = atom_idx
            max_attention = node_attentions[atom_idx]
    return (max_attention_index, max_attention)


def get_connected_neighbor(atom_idx: int, connected_atoms: list[int], mol: Chem.Mol) -> list:
    """
    Get the closest connected neighbor of an atom.

    Parameters
    ----------
    atom_idx : int
        The idx of the atom.
    connected_atoms : list[int]
        The already connected atoms of the subgraph of the molecule.
    mol : Chem.Mol
        The molecule to get the connected neighbor from.

    Returns
    -------
    list[int, int]
        relative atom idx and absolute atom idx of the connected neighbor.
    """
    for b in mol.GetAtomWithIdx(int(atom_idx)).GetBonds():
        if b.GetBeginAtomIdx() in connected_atoms:
            absolut_idx = b.GetBeginAtomIdx()
            relative_idx = connected_atoms.index(absolut_idx)
            return (relative_idx, absolut_idx)
        if b.GetEndAtomIdx() in connected_atoms:
            absolut_idx = b.GetEndAtomIdx()
            relative_idx = connected_atoms.index(absolut_idx)
            return (relative_idx, absolut_idx)
    return None


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
    list[list[atom_features], list[int]]
        The possible atom features for the molecule and the possible atom idxs.
    """
    possible_atom_features = []
    possible_atom_idxs = []
    for atom in connected_atoms:
        for bond in mol.GetAtomWithIdx(atom).GetBonds():
            if bond.GetBeginAtomIdx() not in connected_atoms:
                possible_atom_features.append(
                    atom_features(mol, bond.GetBeginAtomIdx(), connectedTo=(connected_atoms.index(atom), atom))
                )
                possible_atom_idxs.append(bond.GetBeginAtomIdx())
            if bond.GetEndAtomIdx() not in connected_atoms:
                possible_atom_features.append(
                    atom_features(mol, bond.GetEndAtomIdx(), connectedTo=(connected_atoms.index(atom), atom))
                )
                possible_atom_idxs.append(bond.GetEndAtomIdx())
    return possible_atom_features, possible_atom_idxs


def create_new_node_from_develop_node(current_develop_node: develop_node, current_new_node: node):
    """
    Create a new node from a develop node. Used to convert a the raw construction tree to a final tree.

    Parameters
    ----------
    current_develop_node : develop_node
        The develop node to convert.
    current_new_node : node
        The new node to create.
    """
    atom = current_develop_node.atom
    level = current_develop_node.level
    result, std, attention, size = current_develop_node.get_node_result_and_std_and_attention_and_length()
    new_new_node = node(atom, level, result, std, attention, size)
    current_new_node.add_child(new_new_node)
    for child in current_develop_node.children:
        create_new_node_from_develop_node(child, new_new_node)
