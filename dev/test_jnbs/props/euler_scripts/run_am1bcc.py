from rdkit import Chem
from tqdm import tqdm
import sys
import pandas as pd
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField

sdf_test_path = "/cluster/work/igc/mlehner/test142_psi4_rest/sdf_qmugs500_mbis_collect.sdf"
sdf_test = Chem.SDMolSupplier(sdf_test_path, removeHs=False)
ff = ForceField("openff_unconstrained-2.0.0.offxml")


def get_am1Bcc_charges_averaged(mol):
    try:
        molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        charges_tmp = ff.get_partial_charges(molecule)
        charges = charges_tmp.value_in_unit(charges_tmp.unit).tolist()
        return [round(float(item), 5) for item in charges]
    except:
        return [0 for x in range(mol.GetNumAtoms())]


def get_am1Bcc_charges_conf(mol):
    try:
        molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        charges_tmp = molecule.assign_partial_charges("am1bcc", use_conformers=molecule.conformers)
        charges = charges_tmp.value_in_unit(charges_tmp.unit).tolist()
        return [round(float(item), 5) for item in charges]
    except:
        return [0 for x in range(mol.GetNumAtoms())]


def get_am1mulliken_charge(mol):
    try:
        molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        charges_tmp = molecule.assign_partial_charges('"am1-mulliken"', use_conformers=molecule.conformers)
        charges = charges_tmp.value_in_unit(charges_tmp.unit).tolist()
        return [round(float(item), 5) for item in charges]
    except:
        return [0 for x in range(mol.GetNumAtoms())]


if __name__ == "__main__":
    # get mol index from argument
    array_idx = int(sys.argv[1]) - 1
    print("array_idx: ", array_idx)
    array_width = 1000
    start_idx = array_idx * array_width
    end_idx = start_idx + array_width
    print("start_idx: ", start_idx)
    print("end_idx: ", end_idx)
    # calculate charges
    am1Bcc_charges_averaged = []
    am1Bcc_charges_conf = []
    am1Mulliken_charges = []
    elements = []
    atom_indices = []
    mol_indices = []
    dash_indices = []
    for mol_index in tqdm(range(start_idx, end_idx)):
        try:
            mol = sdf_test[mol_index]
            am1Bcc_charges_averaged.append(get_am1Bcc_charges_averaged(mol))
            am1Bcc_charges_conf.append(get_am1Bcc_charges_conf(mol))
            am1Mulliken_charges.append(get_am1mulliken_charge(mol))
            elements.append([x.GetSymbol() for x in mol.GetAtoms()])
            atom_indices.append([x.GetIdx() for x in mol.GetAtoms()])
            mol_indices.append([mol_index for x in mol.GetAtoms()])
            dash_indices.append([mol.GetProp("DASH_IDX") for x in mol.GetAtoms()])
        except Exception as e:
            print(e)
            pass
    # flatten lists
    am1Bcc_charges_averaged_flat = [item for sublist in am1Bcc_charges_averaged for item in sublist]
    am1Bcc_charges_conf_flat = [item for sublist in am1Bcc_charges_conf for item in sublist]
    am1Mulliken_charges_flat = [item for sublist in am1Mulliken_charges for item in sublist]
    elements_flat = [item for sublist in elements for item in sublist]
    atom_indices_flat = [item for sublist in atom_indices for item in sublist]
    mol_indices_flat = [item for sublist in mol_indices for item in sublist]
    dash_indices_flat = [item for sublist in dash_indices for item in sublist]
    # create dataframe
    df = pd.DataFrame(
        {
            "mol_index": mol_indices_flat,
            "dash_index": dash_indices_flat,
            "atom_index": atom_indices_flat,
            "element": elements_flat,
            "am1Bcc_averaged": am1Bcc_charges_averaged_flat,
            "am1Bcc_conf": am1Bcc_charges_conf_flat,
            "am1Mulliken": am1Mulliken_charges_flat,
        }
    )
    # save dataframe
    df.to_csv(f"./charges/test_{array_idx}.csv", index=False)
    print("Done with array", array_idx)
