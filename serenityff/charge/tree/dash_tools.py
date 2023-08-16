import numpy as np
import pandas as pd

from newick import loads, Node


def atoms_to_Name(atom_list: list) -> str:
    ret_str = ""
    for atom in atom_list:
        ret_str += "#"
        ret_str += "_".join([str(x) for x in atom])
    return ret_str


def name_to_atoms(name: str) -> list:
    ret_list = []
    for atom in name.split("#")[1:]:
        ret_list.append(tuple([int(x) for x in atom.split("_")]))
    return ret_list


df_entry_list = []


def add_oldNode_toNewTree(oldNode, newParent, id_counter=0):
    new_node_name = atoms_to_Name(oldNode.atoms)
    atom_type, con_atom, con_type = oldNode.atoms[0]
    df_entry_list.append(
        [
            oldNode.level,
            atom_type,
            con_atom,
            con_type,
            oldNode.result,
            oldNode.stdDeviation,
            oldNode.attention,
            oldNode.count,
        ]
    )
    new_node = Node(new_node_name, comment=str(id_counter))
    newParent.add_descendant(new_node)
    id_counter += 1
    for child in oldNode.children:
        id_counter = add_oldNode_toNewTree(child, new_node, id_counter)
    return id_counter


def get_NewickTree_and_df(oldNode):
    df_entry_list = []
    newick_tree = loads("root;")[0]
    add_oldNode_toNewTree(oldNode, newick_tree)

    columns = ["level", "atom", "con", "conType", "result", "stdDeviation", "attention", "count"]
    dtypesRaw = [np.int8, np.int8, np.int8, np.int8, np.float16, np.float16, np.float16, np.int32]
    dtypes = dict(zip(columns, dtypesRaw))
    df = pd.DataFrame(df_entry_list, columns=columns).astype(dtypes)
    return newick_tree, df
