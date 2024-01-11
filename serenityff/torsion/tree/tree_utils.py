import pandas as pd
from serenityff.torsion.tree_develop.develop_node import DevelopNode
from serenityff.torsion.tree.dash_tree import DASHTorsionTree


def get_data_from_DEV_node(dev_node: DevelopNode):
    # dev_node.update_average()
    atom = dev_node.atom_features
    level = dev_node.level
    (
        size,
        max_attention,
        mean_attention,
        histogram,
    ) = dev_node.get_DASH_data_from_dev_node()
    return (atom, level, max_attention, mean_attention, size, histogram)


def recursive_DEV_node_to_DASH_tree(
    dev_node: DevelopNode, id_counter: int, parent_id: int, tree_storage: list, data_storage: list
):
    # check if tree_storage length is equal to id_counter
    if len(tree_storage) != id_counter:
        print("ERROR: tree_storage length is not equal to id_counter")
        return
    atom, level, max_attention, mean_attention, size, histogram = get_data_from_DEV_node(dev_node)
    atom_type, con_atom, con_type = atom
    tree_storage.append((id_counter, atom_type, con_atom, con_type, mean_attention, []))
    data_storage.append((level, atom_type, con_atom, con_type, max_attention, size, histogram))
    parent_id = id_counter
    for child in dev_node.children:
        id_counter += 1
        tree_storage[parent_id][5].append(id_counter)
        id_counter = recursive_DEV_node_to_DASH_tree(child, id_counter, parent_id, tree_storage, data_storage)
    return id_counter


def get_DASH_tree_from_DEV_tree(dev_root: DevelopNode, tree_folder_path: str = "./") -> DASHTorsionTree:
    tree_storage = {}
    data_storage = {}
    for child in dev_root.children:
        branch_tree_storage = []
        branch_data_storage = []
        recursive_DEV_node_to_DASH_tree(child, 0, 0, branch_tree_storage, branch_data_storage)
        branch_data_df = pd.DataFrame(
            branch_data_storage,
            columns=["level", "atom_type", "con_atom", "con_type", "max_attention", "size", "histogram"],
        )
        child_id = int(child.atom_features[0])
        tree_storage[child_id] = branch_tree_storage
        data_storage[child_id] = branch_data_df
    tree = DASHTorsionTree(tree_folder_path=tree_folder_path, preload=False)
    tree.data_storage = data_storage
    tree.tree_storage = tree_storage
    # print("tree_storage: ", tree_storage)
    # print("data_storage: ", data_storage)
    tree.save_all_trees_and_data()
    return tree
