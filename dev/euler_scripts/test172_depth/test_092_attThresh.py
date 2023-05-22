import sys
from rdkit import Chem

# import numpy as np
import pandas as pd
import time
from serenityff.charge.tree.tree import Tree

tree_pruned_folder = "./tree"
sdf_file_path = "./test.sdf"

# get jobindex as argument
jobindex = int(sys.argv[1])

mol_sup = Chem.SDMolSupplier(sdf_file_path, removeHs=False)
print(len(mol_sup))
test_tree = Tree()
test_tree.from_folder_pickle(tree_pruned_folder)

sys.setrecursionlimit(10000)

# attention_thresholds_to_test = np.linspace(0.3, 1.2, 21)
# att_thresh = attention_thresholds_to_test[jobindex]
print(f"Depth threshold: {jobindex}")
start = time.time()
df_list = []
for mol_idx, mol in enumerate(mol_sup):
    try:
        num_atoms = mol.GetNumAtoms()
        tree_charges_stuff = test_tree.match_molecule_atoms(
            mol=mol, return_raw=True, return_std=True, return_match_depth=True, max_depth=jobindex
        )
        tree_charges = tree_charges_stuff[0]
        tree_raw = tree_charges_stuff[1]
        tree_std = tree_charges_stuff[2]
        tree_depth = tree_charges_stuff[3]
        mbis_charge = [float(x) for x in mol.GetProp("MBIScharge").split("|")]
        if mol.HasProp("DFT:MULLIKEN_CHARGES"):
            mulliken_charge = [float(x) for x in mol.GetProp("DFT:MULLIKEN_CHARGES").split("|")]
        else:
            mulliken_charge = [float(x) for x in mol.GetProp("XTB_MulikenCharge").split("|")]
        atom_idx = list(range(num_atoms))
        mol_idxs = [mol_idx] * num_atoms
        element = [x.GetSymbol() for x in mol.GetAtoms()]
        df_list.append(
            pd.DataFrame(
                {
                    "mol_index": mol_idxs,
                    "atom_index": atom_idx,
                    "element": element,
                    "tree_charge": tree_charges,
                    "mbis_charge": mbis_charge,
                    "mulliken_charge": mulliken_charge,
                    "tree_raw": tree_raw,
                    "tree_std": tree_std,
                    "tree_depth": tree_depth,
                }
            )
        )
    except Exception as e:
        print(e)
        pass
used_time = time.time() - start
df = pd.concat(df_list)
df.to_csv(f"./df_{jobindex}.csv")
print(f"runtime_tree={time.time()-start}")
print("all done")
