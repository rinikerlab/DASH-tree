from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# import time
import sys
import pandas as pd
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from serenityff.charge.tree.tree import Tree

sdf_test_path = "./test.sdf"
sdf_test = Chem.SDMolSupplier(sdf_test_path, removeHs=False)
tree_folder = "../test166_tree/test_009_out/tree/"
ff = ForceField("openff_unconstrained-2.0.0.offxml")

# import tree
test_tree = Tree()
test_tree.from_folder_pickle(tree_folder)


def get_am1Bcc_charges(mol):
    try:
        molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        charges_tmp = ff.get_partial_charges(molecule)
        charges = charges_tmp.value_in_unit(charges_tmp.unit).tolist()
        return [round(float(item), 5) for item in charges]
    except Exception:
        return [0 for x in range(mol.GetNumAtoms())]


def get_gasteiger_charges(mol):
    try:
        AllChem.ComputeGasteigerCharges(mol)
        return [round(float(x.GetProp("_GasteigerCharge")), 5) for x in mol.GetAtoms()]
    except Exception:
        return [0 for x in range(mol.GetNumAtoms())]


def get_mmff_charges(mol):
    try:
        mm = AllChem.MMFFGetMoleculeProperties(mol)
        return [round(float(mm.GetMMFFPartialCharge(x)), 5) for x in range(mol.GetNumAtoms())]
    except Exception:
        return [0 for x in range(mol.GetNumAtoms())]


def get_tree_charges(mol):
    try:
        return test_tree.match_molecule_atoms(mol, attention_threshold=5.6)[0]
    except Exception:
        return [0 for x in range(mol.GetNumAtoms())]


def get_mbis_charges(mol):
    try:
        return [float(x) for x in mol.GetProp("MBIScharge").split("|")]
    except Exception:
        return [0 for x in range(mol.GetNumAtoms())]


if __name__ == "__main__":
    # get mol index from argument
    array_idx = int(sys.argv[1]) - 1
    print("array_idx: ", array_idx)
    array_width = 100
    start_idx = array_idx * array_width
    end_idx = start_idx + array_width
    print("start_idx: ", start_idx)
    print("end_idx: ", end_idx)
    # calculate charges
    am1Bcc_charges = []
    gasteiger_charges = []
    mmff_charges = []
    tree_charges = []
    mbis_charges = []
    elements = []
    atom_indices = []
    mol_indices = []
    for mol_index in tqdm(range(start_idx, end_idx)):
        try:
            mol = sdf_test[mol_index]
            am1Bcc_charges.append(get_am1Bcc_charges(mol))
            gasteiger_charges.append(get_gasteiger_charges(mol))
            mmff_charges.append(get_mmff_charges(mol))
            tree_charges.append(get_tree_charges(mol))
            mbis_charges.append(get_mbis_charges(mol))
            elements.append([x.GetSymbol() for x in mol.GetAtoms()])
            atom_indices.append([x.GetIdx() for x in mol.GetAtoms()])
            mol_indices.append([mol_index for x in mol.GetAtoms()])
        except Exception as e:
            print(e)
            pass
    # flatten lists
    am1Bcc_charges_flat = [item for sublist in am1Bcc_charges for item in sublist]
    gasteiger_charges_flat = [item for sublist in gasteiger_charges for item in sublist]
    mmff_charges_flat = [item for sublist in mmff_charges for item in sublist]
    tree_charges_flat = [item for sublist in tree_charges for item in sublist]
    mbis_charges_flat = [item for sublist in mbis_charges for item in sublist]
    elements_flat = [item for sublist in elements for item in sublist]
    atom_indices_flat = [item for sublist in atom_indices for item in sublist]
    mol_indices_flat = [item for sublist in mol_indices for item in sublist]
    # create dataframe
    df = pd.DataFrame(
        {
            "mol_index": mol_indices_flat,
            "atom_index": atom_indices_flat,
            "element": elements_flat,
            "am1Bcc": am1Bcc_charges_flat,
            "gasteiger": gasteiger_charges_flat,
            "mmff": mmff_charges_flat,
            "tree": tree_charges_flat,
            "mbis": mbis_charges_flat,
        }
    )
    # save dataframe
    df.to_csv(f"./charges/test_{array_idx}.csv", index=False)
    print("Done with array", array_idx)
