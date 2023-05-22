from rdkit import Chem
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from serenityff.charge.tree.tree import Tree
from serenityff.charge.tree_develop.tree_constructor import Tree_constructor


data_file = "./combined.csv"  # "../test133_explain/combined.csv"
# sdf_suply="../test131_psi4_rest/combined.sdf"
sdf_suply = "./combined_multi.sdf"  # "./sdf_explain.sdf"#"../test133_explain/sdf_explain.sdf"
nrows = None
data_split = 0.1
out_folder = "./test_009_out"
split_indices_path = "/cluster/work/igc/mlehner/test154_explain/GNN_lr_0.00010000_batch_64_seed_1_index.csv"

print("00 - starting", flush=True)

tree_constructor = Tree_constructor(
    df_path=data_file,
    sdf_suplier=sdf_suply,
    nrows=nrows,
    num_layers_to_build=16,
    data_split=data_split,
    verbose=True,
    sanitize=False,
    sanitize_charges=False,
    split_indices_path=split_indices_path,
)
print("01 - tree_constructor created", flush=True)

tree_constructor.create_tree_level_0()
print("02 - tree_constructor created level 0", flush=True)

tree_constructor.build_tree(num_processes=6)
print("03 - tree_constructor built tree", flush=True)

tree_constructor.convert_tree_to_node(delDevelop=True)
print("04 - tree_constructor converted tree to node", flush=True)

tree_constructor.new_root.fix_nan_stdDeviation()
print("05 - tree_constructor fixed nan stdDeviation", flush=True)

for num, child in enumerate(tree_constructor.new_root.children):
    child.to_file(f"{out_folder}/tree/tree_{num}.csv")
    with open(f"{out_folder}/tree/tree_{num}.pkl", "wb") as f:
        pickle.dump(child, f)
print("06 - tree_constructor saved tree to file", flush=True)

tree_constructor.test_df.to_csv(out_folder + "/test_df.csv")
print("07 - tree_constructor saved test_df to file", flush=True)

new_tree = Tree()
new_tree.from_folder(out_folder + "/tree", verbose=True)
print("08 - new_tree created from file", flush=True)

for child in new_tree.root.children:
    child.prune()
new_tree.root.fix_nan_stdDeviation()
print("09 - new_tree pruned", flush=True)

for num, child in enumerate(new_tree.root.children):
    child.to_file(f"{out_folder}/tree_pruned/tree_{num}.csv")
    with open(f"{out_folder}/tree/tree_{num}.pkl", "wb") as f:
        pickle.dump(child, f)
print("10 - new_tree saved to file", flush=True)

test_tree = Tree()
test_tree.root = new_tree.root


def calculate_RMSE_tree_vs_truth(df, arg1, arg2):
    return ((df[arg1] - df[arg2]) ** 2).mean() ** 0.5


def calculate_r2_tree_vs_truth(df, arg1, arg2):
    return df[[arg1, arg2]].corr()[arg1][arg2] ** 2


test_df = pd.read_csv(out_folder + "/test_df.csv")


def x_match_molecules_atoms(tree, mol, mol_idx, max_depth=0):
    return_list = []
    mbis_charges = mol.GetProp("MBIScharge").split("|")
    for atom in mol.GetAtoms():
        return_dict = {}
        atom_idx = atom.GetIdx()
        return_dict["mol_idx"] = int(mol_idx)
        return_dict["atom_idx"] = int(atom_idx)
        return_dict["atomtype"] = atom.GetSymbol()
        return_dict["truth"] = float(mbis_charges[atom_idx])
        try:
            result, node_path = tree.match_new_atom(atom_idx, mol, max_depth=max_depth)
            return_dict["tree"] = float(result)
            return_dict["tree_std"] = node_path[-1].stdDeviation
        except Exception as e:
            print(e)
            return_dict["tree"] = np.NAN
            return_dict["tree_std"] = np.NAN
        return_list.append(return_dict)

    # normalize_charge symmetric
    tot_charge_truth = np.round(np.sum([float(x) for x in mbis_charges]))
    tot_charge_tree = np.sum([x["tree"] for x in return_list])
    for x in return_list:
        x["tree_norm1"] = x["tree"] - ((tot_charge_tree - tot_charge_truth) / mol.GetNumAtoms())

    # normalize_charge std weighted
    tot_std = np.sum([x["tree_std"] for x in return_list])
    for x in return_list:
        x["tree_norm2"] = x["tree"] + (tot_charge_truth - tot_charge_tree) * (x["tree_std"] / tot_std)

    return return_list


def x_match_dataset(tree, mol_sup, stop=1000000, max_depth=0):
    i = 0
    tot_list = []
    for mol in tqdm(mol_sup):
        if i >= stop:
            break
        tot_list.extend(x_match_molecules_atoms(tree, mol, i, max_depth=max_depth))
        i += 1
    return pd.DataFrame(tot_list)


def x_match_dataset_with_indices(tree, mol_sup, indices):
    i = 0
    tot_list = []
    for mol in tqdm(mol_sup):
        if i in indices:
            tot_list.extend(x_match_molecules_atoms(tree, mol, i, max_depth=12))
            i += 1
        else:
            i += 1
    return pd.DataFrame(tot_list)


df_test = x_match_dataset_with_indices(
    tree=test_tree, mol_sup=Chem.SDMolSupplier(sdf_suply, removeHs=False), indices=test_df.mol_index.unique().tolist()
)
print("11 - df_test created", flush=True)

print(calculate_RMSE_tree_vs_truth(df_test, "truth", "tree_norm2"))
print("12 - RMSE tree vs truth", flush=True)

df_test.to_csv(out_folder + "/df_tree_matched.csv")
print("13 - df_test saved to file", flush=True)

print("14 - done", flush=True)
