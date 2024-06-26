import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from serenityff.charge.tree.dash_tree import DASHTree

df_props_file = "./props_all.h5"
sdf_file = "./sdf_qmugs500_mbis_collect.sdf"
tree_path = "./dashProps_tree/"

print(
    f"Starting prop script\n df_props_file: {df_props_file}\n sdf_file: {sdf_file}\n tree_path: {tree_path}", flush=True
)

mol_sup = Chem.SDMolSupplier(sdf_file, removeHs=False)
df = pd.read_hdf(df_props_file)

print("Loaded data", flush=True)
mol_sup_idx = -1
last_mol_idx = -1
last_cnf_idx = -1
mol_sup_row = []
for df_line in tqdm(df.itertuples(), total=len(df)):
    mol_idx = df_line.mol_idx
    cnf_idx = df_line.cnf_idx
    if mol_idx != last_mol_idx or cnf_idx != last_cnf_idx:
        mol_sup_idx += 1
        last_mol_idx = mol_idx
        last_cnf_idx = cnf_idx
    mol_sup_row.append(mol_sup_idx)

df["mol_sup_idx"] = mol_sup_row
print("Added mol_sup_idx", flush=True)
tree = DASHTree(tree_folder_path=tree_path)
print("Loaded tree", flush=True)

# select 10% of the data
all_dash_indices = df["DASH_IDX"].unique()
np.random.seed(42)
# selected_dash_indices = set(list(set(np.random.choice(all_dash_indices, int(len(all_dash_indices)*0.1), replace=False)))[:1000])
selected_dash_indices = set(np.random.choice(all_dash_indices, int(len(all_dash_indices) * 0.1), replace=False))
df = df[df["DASH_IDX"].isin(selected_dash_indices)]
mol_sup_idx_to_use = set(df["mol_sup_idx"].unique().tolist())
print("Selected 10% of the data", flush=True)

atomic_dipole_xyz = df[["mbis_dipole_x", "mbis_dipole_y", "mbis_dipole_z"]].values
mbis_data = []
mulliken_data = []
resp1_data = []
resp2_data = []
dual_data = []
mbis_dipole_strength_data = []
dipole_bond_1_data = []
dipole_bond_1_data_reference = []
cnf_avg_hash = []
mol_data = []
dipole_lookup_idx = 0
mol_errors = []
for mol_index, mol in tqdm(enumerate(mol_sup), total=len(mol_sup)):
    if mol_index not in mol_sup_idx_to_use:
        continue
    num_atoms = mol.GetNumAtoms()
    try:
        nodePathList = tree._get_allAtoms_nodePaths(mol=mol)
        mbis = tree.get_molecules_partial_charges(mol, chg_key="result", chg_std_key="std", nodePathList=nodePathList)[
            "charges"
        ]
        mulliken = tree.get_molecules_partial_charges(
            mol, chg_key="mulliken", chg_std_key="std", nodePathList=nodePathList
        )["charges"]
        resp1 = tree.get_molecules_partial_charges(mol, chg_key="resp1", chg_std_key="std", nodePathList=nodePathList)[
            "charges"
        ]
        resp2 = tree.get_molecules_partial_charges(mol, chg_key="resp2", chg_std_key="std", nodePathList=nodePathList)[
            "charges"
        ]
        mbis_dipole_strength = []
        dipole_bond_1 = []
        atom_indices = []
        dash_idx = mol.GetProp("DASH_IDX")
        for atom_idx in range(mol.GetNumAtoms()):
            atom_indices.append(atom_idx)
            nodePath = nodePathList[atom_idx]
            dual_data.append(tree.get_property_noNAN(matched_node_path=nodePath, property_name="dual"))
            mbis_dipole_strength.append(
                tree.get_property_noNAN(matched_node_path=nodePath, property_name="mbis_dipole_strength")
            )
            dipole_bond_1.append(tree.get_property_noNAN(matched_node_path=nodePath, property_name="dipole_bond_1"))
            # get ref bond projected dipole
            ref_dipole_xyz = atomic_dipole_xyz[dipole_lookup_idx]
            bond_vectors = tree._get_attention_sorted_neighbours_bondVectors(mol=mol, atom_idx=atom_idx)
            bond_projected_dipole = tree._project_dipole_to_bonds(bond_vectors=bond_vectors, dipole=ref_dipole_xyz)
            dipole_bond_1_data_reference.append(bond_projected_dipole[0])
            cnf_avg_hash.append(f"{dash_idx}_{atom_idx}")
            dipole_lookup_idx += 1
        mol_idx_per_atom = [mol_index] * len(atom_indices)
        mbis_data.extend(mbis)
        mulliken_data.extend(mulliken)
        resp1_data.extend(resp1)
        resp2_data.extend(resp2)
        mbis_dipole_strength_data.extend(mbis_dipole_strength)
        dipole_bond_1_data.extend(dipole_bond_1)

        # mol props
        mol_dipole_no_atomic = tree.get_molecular_dipole_moment(
            mol, chg_std_key="std", add_atomic_dipoles=False, nodePathList=nodePathList, inDebye=True
        )
        mol_dipole_with_atomic = tree.get_molecular_dipole_moment(
            mol,
            chg_std_key="std",
            add_atomic_dipoles=True,
            nodePathList=nodePathList,
            inDebye=True,
            dipole_magnitude_key="mbis_dipole_strength",
        )
        mol_data.append([mol_index, mol_dipole_no_atomic, mol_dipole_with_atomic, dash_idx])
    except Exception:
        mol_errors.append(mol_index)
        nan_list = [np.nan] * num_atoms
        mbis_data.extend(nan_list)
        mulliken_data.extend(nan_list)
        resp1_data.extend(nan_list)
        resp2_data.extend(nan_list)
        dual_data.extend(nan_list)
        mbis_dipole_strength_data.extend(nan_list)
        dipole_bond_1_data.extend(nan_list)
        dipole_bond_1_data_reference.extend(nan_list)
        cnf_avg_hash.extend(nan_list)
        mol_data.append([mol_index, np.nan, np.nan, np.nan])
    # if mol_index > 1000:
    #    break

print(f"mol_errors: {mol_errors}", flush=True)
df["mbis_pred"] = mbis_data
df["mulliken_pred"] = mulliken_data
df["resp1_pred"] = resp1_data
df["resp2_pred"] = resp2_data
df["dual_pred"] = dual_data
df["mbis_dipole_strength_pred"] = mbis_dipole_strength_data
df["dipole_bond_1_pred"] = dipole_bond_1_data
df["dipole_bond_1"] = dipole_bond_1_data_reference
df["cnf_avg_hash"] = cnf_avg_hash

df.to_csv("test_184_atomData.csv")

# group by cnf_avg_hash
try:
    df_grouped = (
        df[
            [
                "cnf_avg_hash",
                "mol_sup_idx",
                "mol_idx",
                "atom_idx",
                "MBIScharge",
                "mbis_pred",
                "mulliken",
                "mulliken_pred",
                "resp1",
                "resp1_pred",
                "resp2",
                "resp2_pred",
                "dual",
                "dual_pred",
                "mbis_dipole_strength",
                "mbis_dipole_strength_pred",
                "dipole_bond_1",
                "dipole_bond_1_pred",
            ]
        ]
        .groupby("cnf_avg_hash")
        .median()
    )
    df_grouped.to_csv("test_184_atomData_grouped.csv")
except Exception as e:
    print(f"Error in groupby: {e}", flush=True)


# save mol data
df_mol = pd.DataFrame(mol_data, columns=["mol_idx", "mol_dipole_no_atomic", "mol_dipole_with_atomic", "DASH_IDX"])
df_mol.to_csv("test_184_molData.csv")


# add mol_dipole_from_mbis_ref
def get_mol_dipole_from_df_grouped_lines(df_lines):
    # use function with df.groupby("mol_sup_idx") to get dipole for each molecule
    # df_lines is a dataframe with lines from the output file
    # returns a list of dipoles
    mol_sup_idx = df_lines["mol_sup_idx"].iloc[0]
    mol = mol_sup[int(mol_sup_idx)]
    vec_sum = np.zeros(3)
    atom_indices = df_lines["atom_idx"].values
    atom_charges = df_lines["MBIScharge"].values
    mbis_dipole_x = df_lines["mbis_dipole_x"].values
    mbis_dipole_y = df_lines["mbis_dipole_y"].values
    mbis_dipole_z = df_lines["mbis_dipole_z"].values
    mbis_dipoles = np.array([mbis_dipole_x, mbis_dipole_y, mbis_dipole_z]).T
    for ai, chg in zip(atom_indices, atom_charges):
        vec_sum += chg * np.array(mol.GetConformer().GetAtomPosition(int(ai)))
    for mbis_dipole in mbis_dipoles:
        vec_sum += mbis_dipole
    return np.linalg.norm(vec_sum)


print("finshed everytinh except for mol_dipole_from_mbis_ref", flush=True)
tqdm.pandas()
tmp_df = df.groupby("mol_sup_idx").progress_apply(get_mol_dipole_from_df_grouped_lines)
df_mol["mol_dipole_from_mbis_ref"] = tmp_df.values
df_mol.to_csv("test_184_molData_withMBIS_ref.csv")

print("Saved data", flush=True)
