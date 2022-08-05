from rdkit import Chem

from serenityff.charge.tree.atom_features import atom_features
from serenityff.charge.tree.node import node
from serenityff.charge.tree_develop.develop_node import develop_node


def create_mol_from_suply(sdf_suply: str, index: int) -> Chem.Mol:
    return Chem.SDMolSupplier(sdf_suply, removeHs=False)[index]


def get_possible_connected_atom(line, layer):
    mol = line["mol"]
    atom_idx = line["idx_in_mol"]

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


def get_possible_connected_new_atom(line, mol):
    possible_atoms = line["connected_atoms"]
    new_atoms = set()
    for atom in possible_atoms:
        for bond in mol.GetAtomWithIdx(atom).GetBonds():
            if bond.GetBeginAtomIdx() not in possible_atoms:
                new_atoms.add(bond.GetBeginAtomIdx())
            if bond.GetEndAtomIdx() not in possible_atoms:
                new_atoms.add(bond.GetEndAtomIdx())
    return list(new_atoms)


def get_connected_atom_with_max_attention(line, layer):
    node_attentions = line["node_attentions"]
    possible_atoms = get_possible_connected_atom(line, layer)
    max_attention_index = None
    max_attention = 0
    for atom_idx in possible_atoms:
        if node_attentions[atom_idx] > max_attention:
            max_attention_index = atom_idx
            max_attention = node_attentions[atom_idx]
    return (max_attention_index, max_attention)


def get_connected_neighbor(line, idx, mol):
    connected_idx = line["connected_atoms"]
    for b in mol.GetAtomWithIdx(int(idx)).GetBonds():
        if b.GetBeginAtomIdx() in connected_idx:
            absolut_idx = b.GetBeginAtomIdx()
            relative_idx = connected_idx.index(absolut_idx)
            return (relative_idx, absolut_idx)
        if b.GetEndAtomIdx() in connected_idx:
            absolut_idx = b.GetEndAtomIdx()
            relative_idx = connected_idx.index(absolut_idx)
            return (relative_idx, absolut_idx)
    return None


def get_possible_atom_features(mol, connected_atoms):
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
    atom = current_develop_node.atom
    level = current_develop_node.level
    result, std, attention, size = current_develop_node.get_node_result_and_std_and_attention_and_length()
    new_new_node = node(atom, level, result, std, attention, size)
    current_new_node.add_child(new_new_node)
    for child in current_develop_node.children:
        create_new_node_from_develop_node(child, new_new_node)
