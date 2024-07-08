import dftd4
from dftd4 import interface
import numpy as np
import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from tqdm import tqdm

print("start dftd4 calculation", flush=True)

mol_sup = Chem.SDMolSupplier("combined_multi.sdf", removeHs=False)
sd_writer = Chem.SDWriter("mols_comb_dftd4.sdf")

damp_params = interface.DampingParam(method="tpssh")

for mol in tqdm(mol_sup, total=len(mol_sup)):
    atom_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atom_positions = mol.GetConformer().GetPositions()
    atom_positions_bohr = atom_positions * 1.88973
    model = interface.DispersionModel(atom_numbers, atom_positions_bohr)
    res = model.get_dispersion(damp_params, grad=False)
    res.update(**model.get_properties())
    c6_coef = res.get("c6 coefficients")
    c6_coef_diag = np.diag(c6_coef)
    polarizability = res.get("polarizibilities")
    mol.SetProp("DFTD4:C6", "|".join([f"{x:.4f}" for x in c6_coef_diag]))
    mol.SetProp("DFTD4:polarizability", "|".join([f"{x:.4f}" for x in polarizability]))
    sd_writer.write(mol)

print("all dftd4 done", flush=True)
