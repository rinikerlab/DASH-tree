from rdkit import Chem
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

# from pandas.core.common import SettingWithCopyWarning
from pandas.errors import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

origin_folder1 = "/cluster/work/igc/mlehner/test182_psi4_qmugs500/"
origin_folder2 = "/cluster/work/igc/mlehner/test181_psi4_rest/"

error_dict = {"noMBIS": 0, "elem": 0, "std": 0, "errorCheck": 0, "noFile": 0, "noProps": 0, "noDipole": 0}


def get_mbis_charge(mol):
    try:
        mbis_charges = [float(x) for x in mol.GetProp("MBIS_CHARGES").split("|")]
    except:
        try:
            mbis_charges = [float(x) for x in mol.GetProp("MBIScharge").split("|")]
        except:
            mbis_charges = None
    if len(mbis_charges) != mol.GetNumAtoms():
        return None
    return mbis_charges


def check_if_charges_match_element(mbis_charges, elements):
    check_charge_dict_temp = {
        "H": (-0.01, 1.01),
        "C": (-2, 3),
        "N": (-3.5, 3),
        "O": (-4, 3),
        "S": (-10, 10),
        "P": (-10, 10),
        "F": (-10, 0.01),
        "Cl": (-10, 0.01),
        "Br": (-10, 0.01),
        "I": (-10, 0.01),
    }
    check_charge_dict = defaultdict(lambda: (-10, 10), check_charge_dict_temp)
    for chg, element in zip(mbis_charges, elements):
        lower_bound, upper_bound = check_charge_dict[element]
        if chg < lower_bound or chg > upper_bound:
            return False
    return True


def check_if_calculation_was_correct(mol_sup_out, error_type_dict={}, chg_diff_threshold=0.4):
    return_mols = []
    try:
        mbis_charge_data = []
        single_check_mol_idx_valid = []
        num_confs = len(mol_sup_out)
        num_atoms = mol_sup_out[0].GetNumAtoms()
        for i in range(num_confs):
            mbis_charges = get_mbis_charge(mol_sup_out[i])
            if mbis_charges is None:
                error_type_dict["noMBIS"] = error_type_dict.get("noMBIS", 0) + 1
                break
            else:
                # mbis_charge_data.append(mbis_charges)
                if not check_if_charges_match_element(mbis_charges, [x.GetSymbol() for x in mol_sup_out[i].GetAtoms()]):
                    error_type_dict["elem"] = error_type_dict.get("elem", 0) + 1
                    break  # return False
                else:
                    mbis_charge_data.append(mbis_charges)
                    single_check_mol_idx_valid.append(i)
        # check charge difference - case 1, 3 valid charges, case 2 two sets of charges are valid
        if len(mbis_charge_data) == 3:
            for i in range(num_atoms):
                # chg_std = np.std([x[i] for x in mbis_charge_data])
                chg_diff12 = np.abs(mbis_charge_data[0][i] - mbis_charge_data[1][i])
                chg_diff13 = np.abs(mbis_charge_data[0][i] - mbis_charge_data[2][i])
                chg_diff23 = np.abs(mbis_charge_data[1][i] - mbis_charge_data[2][i])
                if (
                    chg_diff12 > chg_diff_threshold
                    and chg_diff13 > chg_diff_threshold
                    and chg_diff23 < chg_diff_threshold
                ):
                    return_mols = [[1, mol_sup_out[1]], [2, mol_sup_out[2]]]
                    error_type_dict["std"] = error_type_dict.get("std", 0) + 1
                elif (
                    chg_diff12 > chg_diff_threshold
                    and chg_diff13 < chg_diff_threshold
                    and chg_diff23 > chg_diff_threshold
                ):
                    return_mols = [[0, mol_sup_out[0]], [2, mol_sup_out[2]]]
                    error_type_dict["std"] = error_type_dict.get("std", 0) + 1
                elif (
                    chg_diff12 < chg_diff_threshold
                    and chg_diff13 > chg_diff_threshold
                    and chg_diff23 > chg_diff_threshold
                ):
                    return_mols = [[0, mol_sup_out[0]], [1, mol_sup_out[1]]]
                    error_type_dict["std"] = error_type_dict.get("std", 0) + 1
                elif (
                    chg_diff12 < chg_diff_threshold
                    and chg_diff13 < chg_diff_threshold
                    and chg_diff23 < chg_diff_threshold
                ):
                    return_mols = [[0, mol_sup_out[0]], [1, mol_sup_out[1]], [2, mol_sup_out[2]]]
                else:
                    error_type_dict["std"] = error_type_dict.get("std", 0) + 1
                    return_mols = []
        elif len(mbis_charge_data) == 2:
            single_mol_valid_idx_1, single_mol_valid_idx_2 = single_check_mol_idx_valid
            for i in range(num_atoms):
                chg_diff = np.abs(mbis_charge_data[0][i] - mbis_charge_data[1][i])
                if chg_diff < chg_diff_threshold:
                    return_mols = [
                        [single_mol_valid_idx_1, mol_sup_out[single_mol_valid_idx_1]],
                        [single_mol_valid_idx_2, mol_sup_out[single_mol_valid_idx_2]],
                    ]
                    error_type_dict["std"] = error_type_dict.get("std", 0) + 1
                else:
                    error_type_dict["std"] = error_type_dict.get("std", 0) + 1
                    return_mols = []
        elif len(mbis_charge_data) == 1:  # no charge difference check available for single conformer
            return_mols = mol_sup_out[[0, single_check_mol_idx_valid[0]]]
        return return_mols
    except Exception as e:
        error_type_dict["errorCheck"] = error_type_dict.get("errorCheck", 0) + 1
        print(e)
        return False


def get_wfn_props(mol_jb_number, origin_folder, sub_folder="props"):
    try:
        hdf_key = f"props_{mol_jb_number}"
        hdf_file = f"{origin_folder}/{sub_folder}/props_{mol_jb_number % 1000}.h5"
        df = pd.read_hdf(hdf_file, key=hdf_key)
        return df
    except:
        return None


def get_props_for_specific_conformer(df, conf_idx):
    num_atoms = df["atom_idx"].max() + 1
    df.reset_index(inplace=True, drop=True)
    df["cnf_idx"] = df.index // num_atoms
    return df[df["cnf_idx"] == conf_idx]


def get_dipole_strength(mol_jb_number, origin_folder, sub_folder="out_dat"):
    file_name = f"{origin_folder}/{sub_folder}/psi4_out_{mol_jb_number}.dat"
    try:
        file_str = ""
        with open(file_name, "r") as f:
            file_str = f.read()
        chunks = file_str.split("MBIS Dipoles")
        chunks = chunks[1:]
        dipole_chunks = [x.split("MBIS Quadrupoles")[0] for x in chunks]
        data = []
        for conf_idx, mbis_str in enumerate(dipole_chunks):
            for i in mbis_str.split("\n")[2:-2]:
                try:
                    center, symbol, atom_num, x, y, z = i.strip().split()
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    dipole_strength = np.sqrt(x**2 + y**2 + z**2)
                    data.append([conf_idx, int(center), symbol.strip(), dipole_strength, x, y, z])
                except:
                    data.append([np.nan, np.nan, np.nan, np.nan])
        return data
    except:
        return None


def get_mol_dipole(mol_jb_number, folder):
    file_path = f"{folder}/out_dat/psi4_out_{mol_jb_number}.dat"
    try:
        with open(file_path, "r") as f:
            file_str = f.read()
            chunks = file_str.split("Magnitude           :")
            chunks = chunks[1:]
            dipoles = []
            for chunk in chunks:
                dipole_line = chunk.split("\n")[0]
                dipole_line.strip()
                dipoles.append(float(dipole_line))
            return dipoles
    except:
        return None


homoLumo_dict_q = {}
homoLumo_dict = {}


def fix_homoLumo(df_row, homo=True):
    if df_row["DASH_IDX"].startswith("QMUGS500"):
        mol_idx = int(df_row["mol_idx"])
        cnf_idx = int(df_row["cnf_idx"])
        dict_entry = homoLumo_dict_q[mol_idx][cnf_idx]
        if homo:
            return dict_entry["homo"]
        else:
            return dict_entry["lumo"]
    else:
        mol_idx = int(df_row["mol_idx"])
        cnf_idx = int(df_row["cnf_idx"])
        dict_entry = homoLumo_dict[mol_idx][cnf_idx]
        if homo:
            return dict_entry["homo"]
        else:
            return dict_entry["lumo"]


def get_homoLumoValues(origin_folder, storage_dict):
    for i in range(1, 2001):
        try:
            df = pd.read_hdf(f"{origin_folder}/homoLumo2/homoLumo_{i}.hdf")
            for index, row in df.iterrows():
                mol_idx = row["mol_idx"]
                cnf_idx = row["cnf_idx"]
                homo = row["homo"]
                lumo = row["lumo"]
                if storage_dict.get(mol_idx) is None:
                    storage_dict[mol_idx] = {cnf_idx: {"homo": homo, "lumo": lumo}}
                else:
                    storage_dict[mol_idx][cnf_idx] = {"homo": homo, "lumo": lumo}
        except:
            pass


def fix_dual_descriptor(origin_folder, mol_jb_number):
    dual_pickle_file = f"{origin_folder}/dual/dual_{mol_jb_number}.pkl"
    x = None
    try:
        x = pickle.load(open(dual_pickle_file, "rb"))
    except:
        pass
    return x


if __name__ == "__main__":
    collect_writer = Chem.SDWriter("./sdf_qmugs500_mbis_collect.sdf")
    all_prop_entries = []
    for i in range(1, 190000):
        if i % 1000 == 0:
            print(f"{datetime.now()} - {i}", flush=True)
        try:
            mol_sup = Chem.SDMolSupplier(f"{origin_folder1}/out_sdf/sdf_mbis_{i}.sdf", removeHs=False)
            mols = check_if_calculation_was_correct(mol_sup)
            props_data = get_wfn_props(i, origin_folder1)  # get props data
            if props_data is None:
                error_dict["noProps"] = error_dict.get("noProps", 0) + 1
                continue
            mbis_dipole_data = get_dipole_strength(i, origin_folder1)  # get dipole strength data
            if mbis_dipole_data is None:
                error_dict["noDipole"] = error_dict.get("noDipole", 0) + 1
                continue
            mol_dipole_data = get_mol_dipole(i, origin_folder1)  # get mol dipole data
            dual_data = fix_dual_descriptor(origin_folder1, i)  # get dual descriptor data
            for conf_idx, mol in mols:
                mol.SetProp("DASH_IDX", f"QMUGS500_{i}")
                mbis_dipole_data_cnf = [x[3] for x in mbis_dipole_data if x[0] == conf_idx]
                mbis_dipole_data_cnf_x = [x[4] for x in mbis_dipole_data if x[0] == conf_idx]
                mbis_dipole_data_cnf_y = [x[5] for x in mbis_dipole_data if x[0] == conf_idx]
                mbis_dipole_data_cnf_z = [x[6] for x in mbis_dipole_data if x[0] == conf_idx]
                props_data_cnf = get_props_for_specific_conformer(props_data, conf_idx)
                try:
                    props_data_cnf["mbis_dipole_strength"] = mbis_dipole_data_cnf
                    props_data_cnf["mbis_dipole_x"] = mbis_dipole_data_cnf_x
                    props_data_cnf["mbis_dipole_y"] = mbis_dipole_data_cnf_y
                    props_data_cnf["mbis_dipole_z"] = mbis_dipole_data_cnf_z
                    props_data_cnf["mol_dipole"] = [mol_dipole_data[conf_idx]] * len(mbis_dipole_data_cnf)
                except:
                    props_data_cnf["mbis_dipole_strength"] = np.nan
                    props_data_cnf["mbis_dipole_x"] = np.nan
                    props_data_cnf["mbis_dipole_y"] = np.nan
                    props_data_cnf["mbis_dipole_z"] = np.nan
                    props_data_cnf["mol_dipole"] = np.nan
                try:
                    props_data_cnf["dual"] = dual_data[conf_idx]
                except:
                    props_data_cnf["dual"] = np.nan
                props_data_cnf["DASH_IDX"] = f"QMUGS500_{i}"
                props_data_cnf["MBIScharge"] = [float(x) for x in mol.GetProp("MBIScharge").split("|")]
                all_prop_entries.append(props_data_cnf)
                collect_writer.write(mol)
        except:
            pass
    for i in range(1, 190000):
        if i % 1000 == 0:
            print(f"{datetime.now()} - {i}", flush=True)
        try:
            mol_sup = Chem.SDMolSupplier(f"{origin_folder2}/out_sdf/sdf_mbis_{i}.sdf", removeHs=False)
            mols = check_if_calculation_was_correct(mol_sup)
            props_data = get_wfn_props(i, origin_folder2)  # get props data
            if props_data is None:
                error_dict["noProps"] = error_dict.get("noProps", 0) + 1
                continue
            mbis_dipole_data = get_dipole_strength(i, origin_folder2)  # get dipole strength data
            if mbis_dipole_data is None:
                error_dict["noDipole"] = error_dict.get("noDipole", 0) + 1
                continue
            mol_dipole_data = get_mol_dipole(i, origin_folder2)  # get mol dipole data
            dual_data = fix_dual_descriptor(origin_folder2, i)  # get dual descriptor data
            for conf_idx, mol in mols:
                mol.SetProp("DASH_IDX", f"Rest_{i}")
                mol.SetProp("MBIScharge", mol.GetProp("MBIS_CHARGES"))
                mbis_dipole_data_cnf = [x[3] for x in mbis_dipole_data if x[0] == conf_idx]
                mbis_dipole_data_cnf_x = [x[4] for x in mbis_dipole_data if x[0] == conf_idx]
                mbis_dipole_data_cnf_y = [x[5] for x in mbis_dipole_data if x[0] == conf_idx]
                mbis_dipole_data_cnf_z = [x[6] for x in mbis_dipole_data if x[0] == conf_idx]
                props_data_cnf = get_props_for_specific_conformer(props_data, conf_idx)
                try:
                    props_data_cnf["mbis_dipole_strength"] = mbis_dipole_data_cnf
                    props_data_cnf["mbis_dipole_x"] = mbis_dipole_data_cnf_x
                    props_data_cnf["mbis_dipole_y"] = mbis_dipole_data_cnf_y
                    props_data_cnf["mbis_dipole_z"] = mbis_dipole_data_cnf_z
                    props_data_cnf["mol_dipole"] = [mol_dipole_data[conf_idx]] * len(mbis_dipole_data_cnf)
                except:
                    props_data_cnf["mbis_dipole_strength"] = np.nan
                    props_data_cnf["mbis_dipole_x"] = np.nan
                    props_data_cnf["mbis_dipole_y"] = np.nan
                    props_data_cnf["mbis_dipole_z"] = np.nan
                    props_data_cnf["mol_dipole"] = np.nan
                try:
                    props_data_cnf["dual"] = dual_data[conf_idx]
                except:
                    props_data_cnf["dual"] = np.nan
                props_data_cnf["DASH_IDX"] = f"Rest_{i}"
                props_data_cnf["MBIScharge"] = [float(x) for x in mol.GetProp("MBIScharge").split("|")]
                all_prop_entries.append(props_data_cnf)
                collect_writer.write(mol)
        except:
            pass
    collect_writer.close()
    print(f"{datetime.now()} - printing error dict\n ============== \n {error_dict} \n ============== \n", flush=True)
    print(f"{datetime.now()} - concat all props", flush=True)
    all_prop_df = pd.concat(all_prop_entries)
    # fix homoLumo values
    print(f"{datetime.now()} - fixing homoLumo values", flush=True)
    homoLumo_dict_q = {}
    homoLumo_dict = {}
    get_homoLumoValues(origin_folder1, homoLumo_dict_q)
    get_homoLumoValues(origin_folder2, homoLumo_dict)
    all_prop_df["homo"] = all_prop_df.apply(lambda x: fix_homoLumo(x), axis=1)
    all_prop_df["lumo"] = all_prop_df.apply(lambda x: fix_homoLumo(x, homo=False), axis=1)
    all_prop_df.to_hdf("./props_all.h5", key="df")
    print(f"{datetime.now()} - all jobs done", flush=True)
    print(error_dict)
