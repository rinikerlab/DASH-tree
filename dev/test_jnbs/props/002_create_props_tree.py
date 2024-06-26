# %%
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from rdkit import Chem
from serenityff.charge.tree.dash_tools import init_neighbor_dict
from serenityff.charge.tree.dash_tree import DASHTree
from tqdm import tqdm

# %%
plt.rcParams.update({"font.size": 16})

# %%
main_folder = "./props/"
df_props_file = f"{main_folder}props_all.h5"
sdf_file = f"{main_folder}sdf_qmugs500_mbis_collect.sdf"
node_path_file = f"{main_folder}node_path_storage_default.pkl"
# tree_path = f"{main_folder}tree/"

# %%
mol_sup = Chem.SDMolSupplier(sdf_file, removeHs=False)

# %%
tree = DASHTree()  # tree_folder_path=tree_path)

# %%
df_props = pd.read_hdf(df_props_file, key="df")

# %%
df_props.head(2)

# %%
node_path_storage = pickle.load(open(node_path_file, "rb"))

# %%
unique_dash_idx = df_props["DASH_IDX"].unique()
num_dash_idx = len(unique_dash_idx)

# %%
num_qmugs = 0
for dash_idx in unique_dash_idx:
    if dash_idx.startswith("QMUGS500_"):
        num_qmugs += 1

# %%
props_key_toAdd = [
    "mulliken",
    "resp1",
    "resp2",
    "dual",
    "mbis_dipole_strength",
    "dipole_bond_1",
    "dipole_bond_2",
    "dipole_bond_3",
]
for key in tree.data_storage.keys():
    for propKey in props_key_toAdd:
        tree.data_storage[key][propKey] = np.nan

# %%
dash_idx_and_cnf_idx_2_mol_dict = {}
last_dash_idx = ""
last_cnf_idx = 0
for i, mol in tqdm(enumerate(mol_sup), total=len(mol_sup)):
    dash_idx = mol.GetProp("DASH_IDX")
    if last_dash_idx != dash_idx:
        last_cnf_idx = 0
        dash_idx_and_cnf_idx_2_mol_dict[dash_idx] = {last_cnf_idx: i}
        last_dash_idx = dash_idx
    else:
        last_cnf_idx += 1
        dash_idx_and_cnf_idx_2_mol_dict[dash_idx][last_cnf_idx] = i


# %%
def get_mol_i_from_dash_idx_and_cnf_idx(dash_idx, cnf_idx):
    try:
        return dash_idx_and_cnf_idx_2_mol_dict[dash_idx][cnf_idx]
    except Exception:
        return None


# %%
df_props.head(20)

# %%
# select 10% of the data as validation set
all_dash_indices = df_props["DASH_IDX"].unique()
np.random.seed(42)
selected_dash_indices = set(np.random.choice(all_dash_indices, int(len(all_dash_indices) * 0.1), replace=False))
df_props = df_props[~df_props["DASH_IDX"].isin(selected_dash_indices)]

# %%
data_storage = {}
iter = 0
last_mol_dash_idx = ""
last_neighbor_dict = None
last_bond_vectors_dict = None

for line in tqdm(df_props.itertuples(), total=len(df_props)):
    try:
        dashIdx = line.DASH_IDX
        atomIdx = line.atom_idx
        mol_idx = get_mol_i_from_dash_idx_and_cnf_idx(dashIdx, int(line.cnf_idx))
        node_path = node_path_storage[dashIdx][int(atomIdx)]
        branch_idx = node_path[0]
        if branch_idx not in data_storage.keys():
            data_storage[branch_idx] = {}
        branch_dict = data_storage[branch_idx]
        if last_dash_idx != dashIdx:
            last_dash_idx = dashIdx
            neighbor_dict = init_neighbor_dict(mol_sup[mol_idx])
            last_bond_vectors_dict = {}
        if atomIdx not in last_bond_vectors_dict.keys():
            bond_vectors = tree._get_attention_sorted_neighbours_bondVectors(
                mol_sup[mol_idx], atomIdx, neighbor_dict=neighbor_dict
            )
            last_bond_vectors_dict[atomIdx] = bond_vectors
        else:
            bond_vectors = last_bond_vectors_dict[atomIdx]
        dipole_vec = np.array([line.mbis_dipole_x, line.mbis_dipole_y, line.mbis_dipole_z])
        dipole_bond_1, dipole_bond_2, dipole_bond_3 = tree._project_dipole_to_bonds(
            bond_vectors=bond_vectors, dipole=dipole_vec
        )
        for node_idx in node_path[1:]:
            try:
                if node_idx in branch_dict.keys():
                    node_storage = branch_dict[node_idx]
                    node_storage["mulliken"].append(line.mulliken)
                    node_storage["resp1"].append(line.resp1)
                    node_storage["resp2"].append(line.resp2)
                    node_storage["dual"].append(line.dual)
                    node_storage["mbis_dipole_strength"].append(line.mbis_dipole_strength)
                    node_storage["dipole_bond_1"].append(dipole_bond_1)
                    node_storage["dipole_bond_2"].append(dipole_bond_2)
                    node_storage["dipole_bond_3"].append(dipole_bond_3)
                else:
                    node_storage = {}
                    node_storage["mulliken"] = [line.mulliken]
                    node_storage["resp1"] = [line.resp1]
                    node_storage["resp2"] = [line.resp2]
                    node_storage["dual"] = [line.dual]
                    node_storage["mbis_dipole_strength"] = [line.mbis_dipole_strength]
                    node_storage["dipole_bond_1"] = [dipole_bond_1]
                    node_storage["dipole_bond_2"] = [dipole_bond_2]
                    node_storage["dipole_bond_3"] = [dipole_bond_3]
                    branch_dict[node_idx] = node_storage
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)
        # print(traceback.format_exc())
    # iter += 1
    # if iter > 1000:
    #    break

# %%
mean_data_storage = {}
for branch_idx in data_storage.keys():
    mean_data_storage[branch_idx] = {}
    for node_idx in data_storage[branch_idx].keys():
        mean_data_storage[branch_idx][node_idx] = {}
        for property_key in data_storage[branch_idx][node_idx].keys():
            mean_data_storage[branch_idx][node_idx][property_key] = np.nanmedian(
                data_storage[branch_idx][node_idx][property_key]
            )

# %%
fail_count = 0
succes_count = 0
for branch_key in tqdm(mean_data_storage.keys()):
    tmp_storage = {}
    for property_key in props_key_toAdd:
        tmp_storage[property_key] = tree.data_storage[branch_key][property_key].to_list()
        # print(f"{property_key} - {len(tmp_storage[property_key])} - {branch_key}")

    for node_key in mean_data_storage[branch_key].keys():
        for property_key in mean_data_storage[branch_key][node_key].keys():
            try:
                tmp_storage[property_key][node_key] = mean_data_storage[branch_key][node_key][property_key]
                succes_count += 1
            except Exception:
                fail_count += 1
                # print(branch_key, node_key, property_key)
                pass
    for property_key in props_key_toAdd:  # mean_data_storage[branch_key][node_key].keys():
        tree.data_storage[branch_key][property_key] = tmp_storage[property_key]
        # tree.data_storage[branch_idx].loc[property_key, node_idx] = mean_data_storage[branch_key][node_key][property_key]

# %%
print(f"fail_count: {fail_count}\nsucces_count: {succes_count}")

# %%
tree.tree_folder_path = f"{main_folder}/tree_props/"
print(tree.tree_folder_path)

# %%
tree.save_all_trees_and_data()
