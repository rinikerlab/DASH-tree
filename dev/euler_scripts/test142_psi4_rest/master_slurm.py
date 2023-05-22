# import psi4
# import sys
from rdkit import Chem

# from rdkit.Chem import AllChem
import numpy as np

# import pandas as pd
import os
import time
from datetime import datetime


def get_mbis_charge(mol):
    try:
        mbis_charges = [float(x) for x in mol.GetProp("MBIS_CHARGES").split("|")]
        if len(mbis_charges) != mol.GetNumAtoms():
            return None
        return mbis_charges
    except Exception:
        return None


def check_if_charges_match_element(mbis_charges, elements):
    for chg, element in zip(mbis_charges, elements):
        if element == "H":
            if not (-1 <= np.abs(chg) <= 1):
                return False
        else:
            if not (-6 <= np.abs(chg) <= 6):
                return False
    return True


def check_if_calculation_was_correct(mol_sup_out, error_type_dict={}):
    try:
        mbis_charge_data = []
        num_confs = len(mol_sup_out)
        num_atoms = mol_sup_out[0].GetNumAtoms()
        for i in range(num_confs):
            mbis_charges = get_mbis_charge(mol_sup_out[i])
            if mbis_charges is None:
                error_type_dict["noMBIS"] = error_type_dict.get("noMBIS", 0) + 1
                return False
            mbis_charge_data.append(mbis_charges)
            if not check_if_charges_match_element(mbis_charges, [x.GetSymbol() for x in mol_sup_out[i].GetAtoms()]):
                error_type_dict["elem"] = error_type_dict.get("elem", 0) + 1
                return False
        for i in range(num_atoms):
            chg_std = np.std([x[i] for x in mbis_charge_data])
            if chg_std > 0.9:
                error_type_dict["std"] = error_type_dict.get("std", 0) + 1
                return False
        return True
    except Exception as e:
        error_type_dict["errorCheck"] = error_type_dict.get("errorCheck", 0) + 1
        print(e)
        return False


def submitting_to_slurm(mol_jb_number):
    slurm_comand = f"sbatch -n 1 --cpus-per-task=8 --time=120:00:00 --job-name='q_{mol_jb_number}' --nodes=1 --mem-per-cpu=8192 --tmp=8000 --output='./slurm/{mol_jb_number}.out' --error='./slurm/{mol_jb_number}.err' --wrap='./worker_slurm.sh {mol_jb_number}'"
    os.system(slurm_comand)


def get_num_slurm_jobs():
    num_slurm_jobs = int(os.popen("squeue -u mlehner | wc -l").read())
    return num_slurm_jobs


def slowly_submitting_to_slurm(mol_jb_number, max_num_jobs=400):
    iter = 0
    while iter < 480:
        if get_num_slurm_jobs() < max_num_jobs:
            submitting_to_slurm(mol_jb_number)
            break
        time.sleep(60)
        iter += 1


if __name__ == "__main__":
    print("started, but sleeping first ...", flush=True)
    # time.sleep(28000)
    # print("finihed sleeping")
    error_dict = {"noMBIS": 0, "elem": 0, "std": 0, "errorCheck": 0, "noFile": 0}
    num_errors = 0
    for mol_jb_number in range(1, 190000):
        if mol_jb_number % 100 == 0:
            print(f"{datetime.now()} - {mol_jb_number/190000*100:.2f}% error={num_errors/190000*100:.2f}%", flush=True)
        # out_sdf_file = f"./outs/sdf_mbis_{mol_jb_number}.sdf"
        try:
            out_sdf_file = f"./outs/sdf_mbis_{mol_jb_number}.sdf"
            mol_sup_out = Chem.SDMolSupplier(out_sdf_file, removeHs=False)
        except Exception:
            error_dict["noFile"] = error_dict.get("noFile", 0) + 1
            num_errors += 1
            slowly_submitting_to_slurm(mol_jb_number)
            continue
        if not check_if_calculation_was_correct(mol_sup_out, error_type_dict=error_dict):
            num_errors += 1
            slowly_submitting_to_slurm(mol_jb_number)
    print(error_dict)
