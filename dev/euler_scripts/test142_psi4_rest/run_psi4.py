import os
import sys
import pandas as pd

# from shutil import rmtree
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

# from rdkit.Chem.Draw.IPythonConsole import drawMol3D
from ppqm import XtbCalculator
import psi4
import time
from datetime import datetime


main_folder = "/cluster/work/igc/mlehner/test131_psi4_rest/"

# df_raw = pd.read_csv(f"{main_folder}/sorted_smiles.csv", index_col=0)
# df_TODO = df_raw[df_raw["Set_ID"] != 0]
smiles_jb = ""
mol_jb_number = 0

optimize_options = {
    "gfn": 2,
    "opt": "tight",
    "alpb": "CHCl3",
    "cycles": 600,
}

psi4.set_memory("8000 MB")
psi4.set_num_threads(8)


psi4_options = {
    "basis": "def2-tzvp",
    "scf_type": "df",
    "d_convergence": 5,
    "e_convergence": 5,
    "maxiter": 600,
    "mbis_maxiter": 600,
    "mbis_d_convergence": 1e-5,
    "mbis_radial_points": 75,
    "mbis_spherical_points": 302,
    "pcm": True,
    "pcm_scf_type": "total",
    "pcm__input": """
                Units = Angstrom
                Medium {
                    SolverType = IEFPCM
                    Solvent = Chloroform
                }
                Cavity {
                    RadiiSet = Bondi
                    Type = GePol
                    Scaling = False
                    Area = 0.3
                    Mode = Implicit
                }
                """,
}
psi4.set_options(psi4_options)


def get_matrix_element_from_flat_matrix(i, j, flat_matrix):
    if i == j:
        return 0
    elif i < j:
        index_in_flat_matrix = int((j * (j - 1) / 2) + i)
        return flat_matrix[index_in_flat_matrix]
    else:
        index_in_flat_matrix = int((i * (i - 1) / 2) + j)
        return flat_matrix[index_in_flat_matrix]


def generate_3_different_conformations(smiles, print_energies=False, max_rms=1, verbose=False):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(
        mol,
        numConfs=100,
        pruneRmsThresh=0.01,
        randomSeed=0xF00D,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        ETversion=2,
    )
    if verbose:
        print(len(cids), flush=True)
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=1000)
    pivot = np.argmin([res[i][1] for i in range(len(res))])
    matrix = AllChem.GetConformerRMSMatrix(mol)
    rms_to_pivot = np.array(
        [get_matrix_element_from_flat_matrix(pivot, i, matrix) for i in range(mol.GetNumConformers())]
    )
    possible_new_indices = np.where(rms_to_pivot > max_rms)[0]
    pivot2 = int(possible_new_indices[np.argmin([res[i][1] for i in possible_new_indices])])
    rms_to_pivot2 = np.array(
        [get_matrix_element_from_flat_matrix(pivot2, i, matrix) for i in range(mol.GetNumConformers())]
    )
    possible_new_indices2_2 = np.where(rms_to_pivot2 > max_rms)[0]
    possible_new_indices2 = np.intersect1d(possible_new_indices, possible_new_indices2_2)
    pivot3 = int(possible_new_indices2[np.argmin([res[i][1] for i in possible_new_indices2])])
    ret_mol = Chem.Mol(mol)
    ret_mol.RemoveAllConformers()
    for i in [pivot, pivot2, pivot3]:
        ret_mol.AddConformer(mol.GetConformer(int(i)), assignId=True)
    if print_energies:
        for i in [pivot, pivot2, pivot3]:
            print(res[i][1])
    return ret_mol


def try_max_rms(smiles, verbose=False):
    max_rms = 4
    failed_to_generate_3_confs = False
    while True:
        try:
            ret_mol = generate_3_different_conformations(smiles, max_rms=max_rms)
            return ret_mol
        except Exception:
            max_rms *= 0.85
            if max_rms < 0.01:
                failed_to_generate_3_confs = True
                break
    # fall back to non different conformations
    if failed_to_generate_3_confs:
        print("failed to generate 3 different conformations", flush=True)
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        cids = AllChem.EmbedMultipleConfs(
            mol, numConfs=3, randomSeed=0xF00D, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, ETversion=2
        )
        res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=1000)
        if verbose:
            print(len(res), len(cids), flush=True)
        return mol


def write_psi4_geometry_string(conf):
    """
    Writes the geometry string for a molecule.
    """
    return_string = ""
    for atom in conf.GetOwningMol().GetAtoms():
        x, y, z = conf.GetAtomPosition(atom.GetIdx())
        return_string += f"{atom.GetSymbol()} {x} {y} {z}\n"
    return return_string


def extract_mbis_charges():
    file_str = ""
    with open(psi4_out_file, "r") as f:
        file_str = f.read()
    mbis_str = file_str.split("MBIS Charges: (a.u.)")[-1].split("MBIS Dipoles: [e a0]")[0]
    data = []
    for i in mbis_str.split("\n")[2:-2]:
        try:
            num, symbol, z, pop, charge = i.strip().split()
            num = int(num)
            symbol = symbol.strip()
            z = int(z)
            pop = float(pop)
            charge = float(charge)
            data.append([num, symbol, z, pop, charge])
        except Exception:
            print(i.strip().split())
    return data


if __name__ == "__main__":
    mol_jb_number = int(sys.argv[1])
    smiles_jb = sys.argv[2]
    psi4_out_file = f"./psi4_out_{mol_jb_number}.dat"
    psi4.core.set_output_file(psi4_out_file, False)
    out_sdf_file = f"./sdf_mbis_{mol_jb_number}.sdf"
    print(f"{datetime.now()} - Job: {mol_jb_number} {smiles_jb}", flush=True)

    mol = try_max_rms(smiles_jb)
    print(f"{datetime.now()} - generated conformations", flush=True)

    xtb = XtbCalculator(cmd="xtb")
    results = xtb.calculate(mol, optimize_options)
    print(f"{datetime.now()} - calculated energies", flush=True)

    for idx, cnf_res in enumerate(results):
        for atom_idx in range(mol.GetNumAtoms()):
            x, y, z = cnf_res["coord"][atom_idx]
            mol.GetConformer(idx).SetAtomPosition(atom_idx, Point3D(x, y, z))
    print(f"{datetime.now()} - set optimized conformer positions", flush=True)

    writer = Chem.SDWriter(out_sdf_file)
    for conf_idx in range(mol.GetNumConformers()):
        mol_string = write_psi4_geometry_string(mol.GetConformer(conf_idx))
        psi4_mol = psi4.geometry(mol_string)
        psi4_mol.set_molecular_charge(0)  # mol.GetFormalCharge())
        psi4_mol.set_multiplicity(1)  # mol.GetNumRadicalElectrons() + 1)
        print(f"{datetime.now()} - Setup done", flush=True)
        energy, wfn = psi4.energy("TPSSh", return_wfn=True)
        print(f"{datetime.now()} - QM done", flush=True)
        time.sleep(10)
        psi4.oeprop(wfn, "MBIS_CHARGES", title="title")
        data_mbis = extract_mbis_charges()
        df_mbis = pd.DataFrame(data_mbis, columns=["num", "symbol", "z", "pop", "charge"])
        # df_mbis.to_csv(f"./outs/psi4_{job_array_os_idx}_step4.csv")
        mbis_charge_prop = "|".join(["{:.4f}".format(x) for x in df_mbis.charge.to_list()])
        mol.SetProp("MBIS_CHARGES", mbis_charge_prop)
        mol.SetProp("MBIS_Energy", "{:.4f}".format(energy))
        mol.SetProp("XTB_Energy", "{:.4f}".format(results[conf_idx]["total_energy"]))
        xtb_muliken_charges = results[conf_idx]["mulliken_charges"]
        xtb_muliken_charge_prop = "|".join(["{:.4f}".format(x) for x in xtb_muliken_charges])
        mol.SetProp("XTB_MulikenCharge", xtb_muliken_charge_prop)
        writer.write(mol, confId=conf_idx)
    writer.close()
    os.remove(psi4_out_file)
    print(f"{datetime.now()} - all jobs done", flush=True)
