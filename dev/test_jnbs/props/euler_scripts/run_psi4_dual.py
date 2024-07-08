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
from scipy.constants import physical_constants
import glob
import pickle


main_folder = "/cluster/work/igc/mlehner/test182_psi4_qmugs500/"
mol_jb_number = 0
psi4.set_memory("50 GB")
psi4.set_num_threads(1)
psi4_options = {
    "basis": "def2-tzvp",
    "scf_type": "df",
    "d_convergence": 5,
    "e_convergence": 5,
    "maxiter": 2000,
    "mbis_maxiter": 2000,
    "mbis_d_convergence": 1e-5,
    "mbis_radial_points": 75,  # 150,#75,
    "mbis_spherical_points": 302,  # 590,#302,
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
    "cubeprop_tasks": ["dual_descriptor"],
}
psi4.set_options(psi4_options)


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
    psi4_out_file = f"./psi4_out_{mol_jb_number}.dat"
    psi4.core.set_output_file(psi4_out_file, False)
    print(f"{datetime.now()} - Job: {mol_jb_number}", flush=True)
    all_dual_lists = []
    for i in range(3):
        try:
            wfn = psi4.core.Wavefunction.from_file(f"{main_folder}/wfn/wfn_{mol_jb_number}_{i}.wfn.npy")
            psi4.cubeprop(wfn)
            cube_file = glob.glob("./DUAL*")[0]
            cube = Cube(cube_file)
            dual_atom_list = cube.get_int_atom_list()
            all_dual_lists.append(dual_atom_list)
            os.remove(cube_file)
            wfn = None
            psi4.core.clean()
            time.sleep(10)
        except Exception as e:
            print(f"{datetime.now()} - Error in {mol_jb_number} {i}", flush=True)
            print(e)
            all_dual_lists.append([np.nan])
    pickle.dump(all_dual_lists, open(f"{main_folder}/dual/dual_{mol_jb_number}.pkl", "wb"))
    print(f"{datetime.now()} - all jobs done", flush=True)
