import pickle
from rdkit import Chem
from serenityff.charge.tree.dash_tree import DASHTree

sdf_file_path = "sdf_qmugs500_mbis_collect.sdf"
mol_sup = Chem.SDMolSupplier(sdf_file_path, removeHs=False)
tree = DASHTree()
# tree = DASHTree(tree_folder_path="/cluster/work/igc/mlehner/test192_DASHtree/")

node_path_storage = {}

last_dash_idx = ""
for mol in mol_sup:
    dash_idx = mol.GetProp("DASH_IDX")
    if dash_idx != last_dash_idx:
        this_mols_node_path_storage = {}
        for atom_idx in range(mol.GetNumAtoms()):
            node_path = tree.match_new_atom(atom=atom_idx, mol=mol)
            this_mols_node_path_storage[atom_idx] = node_path
        node_path_storage[dash_idx] = this_mols_node_path_storage
        last_dash_idx = dash_idx
    else:
        continue

with open("node_path_storage_default.pkl", "wb") as f:
    pickle.dump(node_path_storage, f)

print(f"node_paths saved with {len(node_path_storage)} entries \nfile used {sdf_file_path}, \n{len(mol_sup)} mols")
