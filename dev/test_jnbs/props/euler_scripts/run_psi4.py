import psi4
import resp
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
import glob
import copy

psi4.set_memory("32000 MB")
psi4.set_num_threads(2)
main_folder = "/cluster/work/igc/mlehner/test182_psi4_qmugs500/"

psi4_options = {
    "basis": "def2-tzvp",
    "scf_type": "df",
    "d_convergence": 4,
    "e_convergence": 4,
    "maxiter": 600,
    "mbis_maxiter": 600,
    "mbis_d_convergence": 1e-4,
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


def write_psi4_geometry_string(conf):
    """
    Writes the geometry string for a molecule.
    """
    return_string = "0   1\n"
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
        except:
            print(i.strip().split())
    return data


default_resp_options = {
    "VDW_SCALE_FACTORS": [1.4, 1.6, 1.8, 2.0],
    "VDW_POINT_DENSITY": 1.0,
    "RESP_A": 0.0005,
    "RESP_B": 0.1,
    "RESTRAINT": True,
    "IHFREE": False,
    "WEIGHT": [1],
    "PSI4_OPTIONS": psi4_options,
    "VDW_RADII": {"BR": 1.8},
}


class Cube:
    def __init__(self, fname=None):
        if fname != None:
            try:
                self.read_cube(fname)
            except IOError as e:
                print("File used as input: %s" % fname)
                print("File error ({0}): {1}".format(e.errno, e.strerror))
        else:
            self.default_values()
        return None

    def read_cube(self, fname):
        with open(fname, "r") as fin:
            self.filename = fname
            self.comment1 = fin.readline()  # Save 1st comment
            self.comment2 = fin.readline()  # Save 2nd comment
            nOrigin = fin.readline().split()  # Number of Atoms and Origin
            self.natoms = int(nOrigin[0])  # Number of Atoms
            self.origin = np.array([float(nOrigin[1]), float(nOrigin[2]), float(nOrigin[3])])  # Position of Origin
            nVoxel = fin.readline().split()  # Number of Voxels
            self.NX = int(nVoxel[0])
            self.X = np.array([float(nVoxel[1]), float(nVoxel[2]), float(nVoxel[3])])
            nVoxel = fin.readline().split()  #
            self.NY = int(nVoxel[0])
            self.Y = np.array([float(nVoxel[1]), float(nVoxel[2]), float(nVoxel[3])])
            nVoxel = fin.readline().split()  #
            self.NZ = int(nVoxel[0])
            self.Z = np.array([float(nVoxel[1]), float(nVoxel[2]), float(nVoxel[3])])
            self.atoms = []
            self.atomsXYZ = []
            for atom in range(self.natoms):
                line = fin.readline().split()
                self.atoms.append(line[0])
                self.atomsXYZ.append(list(map(float, [line[2], line[3], line[4]])))
            self.data = np.zeros((self.NX, self.NY, self.NZ))
            i = int(0)
            for s in fin:
                for v in s.split():
                    self.data[int(i / (self.NY * self.NZ)), int((i / self.NZ) % self.NY), int(i % self.NZ)] = float(v)
                    i += 1
            # if i != self.NX*self.NY*self.NZ: raise NameError, "FSCK!"
        return None

    def voxel_position(self, i, j, k):
        origin = self.origin
        voxel_x = origin[0] + (i * np.linalg.norm(self.X))
        voxel_y = origin[1] + (j * np.linalg.norm(self.Y))
        voxel_z = origin[2] + (k * np.linalg.norm(self.Z))
        return np.array([voxel_x, voxel_y, voxel_z])

    def voxel_distance(self, i, j, k, atom):
        vec = self.voxel_position(i, j, k)
        return np.linalg.norm(vec - atom)

    def generate_nearest_neighbour_matrix(self, atoms, max_radius=16):
        closest_atom_mask = np.empty((self.NX, self.NY, self.NZ))
        closest_atom_mask[:] = np.nan
        for i in range(self.NX):
            for j in range(self.NY):
                for k in range(self.NZ):
                    closest_distance = 10000000
                    closest_atom = np.nan
                    for atom_idx, atom in enumerate(atoms):
                        distance = self.voxel_distance(i, j, k, atom)
                        if distance < closest_distance and distance < max_radius:
                            closest_distance = distance
                            closest_atom = atom_idx
                    closest_atom_mask[i, j, k] = closest_atom
        return closest_atom_mask

    def get_int_from_voxels_assigned_to_nearest_atom(self, atoms, max_radius=16, verbose=False):
        closest_atom_mask = self.generate_nearest_neighbour_matrix(atoms)
        return_list = []
        voxelMatrix = np.array([self.X, self.Y, self.Z])
        vol_tot = np.linalg.det(voxelMatrix)
        for atom_idx, atom in enumerate(atoms):
            atom_i_mask = np.ones((self.NX, self.NY, self.NZ)) * (closest_atom_mask == atom_idx)
            if verbose:
                print(np.sum(atom_i_mask))
            masked_data = self.data * atom_i_mask
            vol = vol_tot
            return_list.append(np.sum(masked_data * vol))
        return return_list

    def get_int_atom_list(self):
        return self.get_int_from_voxels_assigned_to_nearest_atom(np.array(self.atomsXYZ))


if __name__ == "__main__":
    mol_jb_number = int(sys.argv[1])
    sdf_mol_sup = Chem.SDMolSupplier(f"./sdfs/{mol_jb_number}.sdf", removeHs=False)
    psi4_out_file = f"./psi4_out_{mol_jb_number}.dat"
    psi4.core.set_output_file(psi4_out_file, False)
    out_sdf_file = f"./sdf_mbis_{mol_jb_number}.sdf"
    cube_file = "XXX"
    all_df_entries_lists = []
    writer = Chem.SDWriter(out_sdf_file)
    for conf_idx, mol in enumerate(sdf_mol_sup):
        try:
            mol_string = write_psi4_geometry_string(mol.GetConformer())
            psi4_mol = psi4.geometry(mol_string)
            # print(f"{datetime.now()} - Setup done", flush=True)
            energy, wfn = psi4.energy("TPSSh", return_wfn=True)
            wfn.to_file(f"./wfn_{mol_jb_number}_{conf_idx}.wfn")
            # print(f"{datetime.now()} - QM done", flush=True)
            time.sleep(10)
            psi4.oeprop(wfn, "MBIS_CHARGES", title="title")
            data_mbis = extract_mbis_charges()
            df_mbis = pd.DataFrame(data_mbis, columns=["num", "symbol", "z", "pop", "charge"])
            mbis_charge_prop = "|".join(["{:.4f}".format(x) for x in df_mbis.charge.to_list()])
            mol.SetProp("MBIScharge", mbis_charge_prop)
            writer.write(mol)
            # get props
            num_atoms = psi4_mol.natom()
            elements = [psi4_mol.symbol(i) for i in range(num_atoms)]
            try:
                homo = wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha()]
                lumo = wfn.epsilon_a_subset("AO", "ALL").np[wfn.nalpha() + 1]
            except:
                homo, lumo = [np.nan, np.nan]
            # get mulliken charges from wfn
            try:
                psi4.oeprop(wfn, "MULLIKEN_CHARGES")
                mulliken_charges = [wfn.atomic_point_charges().np[i] for i in range(psi4_mol.natom())]
            except:
                mulliken_charges = [np.nan] * psi4_mol.natom()
            # get Dual Descriptor from wfn
            try:
                psi4.cubeprop(wfn)
                time.sleep(10)
                cube_file = glob.glob("./DUAL*")[0]
                cube = Cube(cube_file)
                dual_atom_list = cube.get_int_atom_list()
            except:
                dual_atom_list = [np.nan] * psi4_mol.natom()
            # get RESP charges from wfn
            try:
                resp_options = copy.deepcopy(default_resp_options)
                psi4.set_options(resp_options["PSI4_OPTIONS"])
                resp_options["wfn"] = wfn
                charges1 = resp.resp([psi4_mol], resp_options)
                resp_options["RESP_A"] = 0.001
                resp.set_stage2_constraint(psi4_mol, charges1[1], resp_options)
                charges2 = resp.resp([psi4_mol], resp_options)
            except:
                charges1 = [np.nan] * psi4_mol.natom()
                charges2 = [np.nan] * psi4_mol.natom()
            # create dataframe entries
            mol_index_for_df = [mol_jb_number] * len(elements)
            atom_idx_for_df = list(range(len(elements)))
            homo_list = [homo] * len(elements)
            lumo_list = [lumo] * len(elements)
            df = pd.DataFrame(
                {
                    "mol_idx": mol_index_for_df,
                    "atom_idx": atom_idx_for_df,
                    "element": elements,
                    "mulliken": mulliken_charges,
                    "resp1": charges1[1],
                    "resp2": charges2[1],
                    "dual": dual_atom_list,
                    "homo": homo_list,
                    "lumo": lumo_list,
                }
            )
            all_df_entries_lists.append(df)
            # clean up
            try:
                os.remove(cube_file)
            except:
                pass
            # wfn = None
            # psi4.core.clean()
            time.sleep(10)
        except Exception as e:
            print(f"{datetime.now()} - error", flush=True)
            print(e)
    writer.close()
    df_tot = pd.concat(all_df_entries_lists)
    hdf_file_index = mol_jb_number % 1000
    df_tot.to_hdf(f"{main_folder}/props/props_{hdf_file_index}.h5", key=f"props_{mol_jb_number}", mode="a")
    ### Example read prop df:
    # df = pd.read_hdf()
    print(f"{datetime.now()} - all jobs done", flush=True)
