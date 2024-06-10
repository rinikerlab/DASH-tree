import os
import pickle
import gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# from multiprocessing import Process, Manager
# from numba import njit, objmode, types

# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import io

try:
    import IPython.display
except ImportError:
    pass
from PIL import Image
from rdkit.Chem.Draw import rdMolDraw2D
from collections import defaultdict

from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.data import default_dash_tree_path
from serenityff.charge.utils.rdkit_typing import Molecule
from serenityff.charge.tree.dash_tools import (
    new_neighbors,
    new_neighbors_atomic,
    init_neighbor_dict,
)


class DASHTree:
    def __init__(
        self,
        tree_folder_path: str = default_dash_tree_path,
        preload: bool = True,
        verbose: bool = True,
        num_processes: int = 1,
    ) -> None:
        """
        Class to handle DASH trees and data

        Parameters
        ----------
        tree_folder_path : str
            Path to folder containing DASH trees and data
        preload : bool
            If True, load all trees and data into memory, if False, load on demand
        verbose : bool
            If True, print status messages
        num_processes : int
            Number of processes to use for loading and assigning molecules.
            TODO: This is currently slow and not recommended
        """
        self.tree_folder_path = tree_folder_path
        self.verbose = verbose
        self.num_processes = num_processes
        self.tree_storage = {}
        self.data_storage = {}
        self.atom_feature_type = AtomFeatures
        if preload:
            self.load_all_trees_and_data()

    ########################################
    #   Tree import/export functions
    ########################################

    # tree file format:
    # int(id_counter), int(atom_type), int(con_atom), int(con_type), float(oldNode.attention), []children

    def load_all_trees_and_data(self) -> None:
        """
        Load all trees and data from the tree_folder_path, expects files named after the atom feature key and
        the file extension .gz for the tree and .h5 for the data
        Examples:
            0.gz, 0.h5 for the tree and data of the atom feature with key 0
        """
        if self.verbose:
            print("Loading DASH tree data")
        # # import all files
        # if True:  # self.num_processes <= 1:
        for i in range(self.atom_feature_type.get_number_of_features()):
            tree_path = os.path.join(self.tree_folder_path, f"{i}.gz")
            df_path = os.path.join(self.tree_folder_path, f"{i}.h5")
            self.load_tree_and_data(tree_path, df_path, branch_idx=i)
        # else:
        # Threads don't seem to work due to HDFstore key error
        #    with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
        #        for i in range(self.atom_feature_type.get_number_of_features()):
        #            tree_path = os.path.join(self.tree_folder_path, f"{i}.gz")
        #            df_path = os.path.join(self.tree_folder_path, f"{i}.h5")
        #            executor.submit(self.load_tree_and_data, tree_path, df_path, i)
        if self.verbose:
            print(f"Loaded {len(self.tree_storage)} trees and data")

    def load_tree_and_data(self, tree_path: str, df_path: str, hdf_key: str = "df", branch_idx: int = None) -> None:
        """
        Load a tree and data from the tree_folder_path, expects files named after the atom feature key and
        the file extension .gz for the tree and .h5 for the data
        Examples:
            0.gz, 0.h5 for the tree and data of the atom feature with key 0

        Parameters
        ----------
        tree_path : str
            the path to the tree file
        df_path : str
            the path to the data file
        hdf_key : str, optional
            the key of the data in the hdf file, by default "df"
        branch_idx : int, optional
            the atom feature key of the tree branch, by default takes the key from the file name
        """
        if branch_idx is None:
            branch_idx = int(os.path.basename(tree_path).split(".")[0])
        with gzip.open(tree_path, "rb") as f:
            tree = pickle.load(f)
        df = pd.read_hdf(df_path, key=hdf_key, mode="r")
        self.tree_storage[branch_idx] = tree
        self.data_storage[branch_idx] = df

    def save_all_trees_and_data(self) -> None:
        """
        Save all trees and data to the tree_folder_path with the file names {branch_idx}.gz and {branch_idx}.h5
        """
        if self.verbose:
            print(f"Saving DASH tree data to {len(self.tree_storage)} files in {self.tree_folder_path}")
        for branch_idx in tqdm(self.tree_storage):
            self.save_tree_and_data(branch_idx)

    def save_tree_and_data(self, branch_idx: int) -> None:
        """
        Save a tree branch and data to the tree_folder_path with the file names {branch_idx}.gz and {branch_idx}.h5

        Parameters
        ----------
        branch_idx : int
            Atom feature key of the tree branch to save
        """
        tree_path = os.path.join(self.tree_folder_path, f"{branch_idx}.gz")
        df_path = os.path.join(self.tree_folder_path, f"{branch_idx}.h5")
        self._save_tree_and_data(branch_idx, tree_path, df_path)

    def _save_tree_and_data(self, branch_idx: int, tree_path: str, df_path: str) -> None:
        with gzip.open(tree_path, "wb") as f:
            pickle.dump(self.tree_storage[branch_idx], f)
        self.data_storage[branch_idx].to_hdf(df_path, key="df", mode="w")

    ########################################
    #   Tree assignment functions
    ########################################

    def _pick_subgraph_expansion_node(
        self,
        current_node: int,
        branch_idx: int,
        possible_new_atom_features: list,
        possible_new_atom_idxs: list,
    ) -> tuple:
        current_node_children = self.tree_storage[branch_idx][current_node][5]
        for child in current_node_children:
            child_tree_node = self.tree_storage[branch_idx][child]
            child_af = (child_tree_node[1], child_tree_node[2], child_tree_node[3])
            for possible_atom_feature, possible_atom_idx in zip(possible_new_atom_features, possible_new_atom_idxs):
                if possible_atom_feature == child_af:
                    return (child, possible_atom_idx)
        return (None, None)

    def _get_init_layer(self, mol: Molecule, atom: int, max_depth: int):
        init_atom_feature = self.atom_feature_type.atom_features_from_molecule_w_connection_info(mol, atom)
        branch_idx = init_atom_feature[0]  # branch_idx is the key of the AtomFeature without connection info
        matched_node_path = [branch_idx, 0]
        # Special case for H -> only connect to heavy atom and ignore H
        if mol.GetAtomWithIdx(atom).GetAtomicNum() == 1:
            h_connected_heavy_atom = mol.GetAtomWithIdx(atom).GetNeighbors()[0].GetIdx()
            init_atom_feature = self.atom_feature_type.atom_features_from_molecule_w_connection_info(
                mol, h_connected_heavy_atom
            )
            child, _ = self._pick_subgraph_expansion_node(0, branch_idx, [init_atom_feature], [h_connected_heavy_atom])
            matched_node_path.append(child)
            atom_indices_in_subgraph = [h_connected_heavy_atom]  # skip Hs as they are only treated implicitly
            max_depth -= 1  # reduce max_depth by 1 as we already added one node
        else:
            atom_indices_in_subgraph = [atom]
        return branch_idx, matched_node_path, atom_indices_in_subgraph, max_depth

    def match_new_atom(
        self,
        atom: int,
        mol: Molecule,
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_increment_threshold: float = 0,
        return_atom_indices: bool = False,
        neighbor_dict=None,
    ):
        """
        Match a atom in a molecule to a DASH tree subgraph. The matching is done by starting at the atom and
        traversing the tree until the max_depth is reached or the attention_threshold is exceeded.
        If the attention_increment_threshold is exceeded, the matching is stopped and the current path is returned.

        Parameters
        ----------
        atom : int
            Atom index in the molecule of the atom to match
        mol : Molecule
            RDKit molecule object in which the atom is located
        max_depth : int
            Maximum depth of the tree to traverse
        attention_threshold : float
            Maximum cumulative attention value to traverse the tree
        attention_increment_threshold : float
            Minimum attention increment to stop the traversal
        """
        if neighbor_dict is None:
            neighbor_dict = init_neighbor_dict(mol, af=self.atom_feature_type)

        # get layer 0, and init all relevant containers
        branch_idx, matched_node_path, atom_indices_in_subgraph, max_depth = self._get_init_layer(
            mol=mol, atom=atom, max_depth=max_depth
        )

        # if data for branch is not preloaded, load it now
        if branch_idx not in self.tree_storage:
            self.load_tree_and_data(
                os.path.join(self.tree_folder_path, f"{branch_idx}.gz"),
                os.path.join(self.tree_folder_path, f"{branch_idx}.h5"),
            )

        # start main DASH loop, expanding the inital subgraph, following the attention
        cummulative_attention = 0
        if max_depth <= 1:
            return matched_node_path
        else:
            for _ in range(1, max_depth):
                possible_new_atom_features, possible_new_atom_idxs = new_neighbors(
                    neighbor_dict, atom_indices_in_subgraph
                )
                child, atom = self._pick_subgraph_expansion_node(
                    matched_node_path[-1],
                    branch_idx,
                    possible_new_atom_features,
                    possible_new_atom_idxs,
                )
                if child is None:
                    break
                matched_node_path.append(child)
                atom_indices_in_subgraph.append(atom)
                node_attention = self.tree_storage[branch_idx][child][4]
                cummulative_attention += node_attention
                possible_new_atom_features_toAdd, possible_new_atom_idxs_toAdd = new_neighbors_atomic(
                    neighbor_dict, atom_indices_in_subgraph, atom
                )
                possible_new_atom_features.extend(possible_new_atom_features_toAdd)
                possible_new_atom_idxs.extend(possible_new_atom_idxs_toAdd)
                if cummulative_attention > attention_threshold:
                    break
                if node_attention < attention_increment_threshold:
                    break
            if return_atom_indices:
                return matched_node_path, atom_indices_in_subgraph
            return matched_node_path

    def get_atom_properties(self, matched_node_path: list = None, mol: Molecule = None, atom: int = None):
        """
        Get the properties of a atom from a matched DASH tree subgraph (node path) or from a molecule
        and atom index (don't use both (mol +atom) and matched_node_path at the same time)

        Parameters
        ----------
        matched_node_path : list
            List of node ids of the matched subgraph (node path) in the order of the traversal

        Returns
        -------
        pd.Series
            All properties of the atom which where stored in the DASH tree
        """
        if matched_node_path is None:
            if mol is None or atom is None:
                raise ValueError("Either matched_node_path or mol + atom must be provided")
            matched_node_path = self.match_new_atom(atom=atom, mol=mol)
        branch_idx = matched_node_path[0]
        atom = matched_node_path[-1]
        if branch_idx not in self.data_storage:
            try:
                self.load_tree_and_data(
                    os.path.join(self.tree_folder_path, f"{branch_idx}.gz"),
                    os.path.join(self.tree_folder_path, f"{branch_idx}.h5"),
                )
            except Exception as e:
                print(f"Error loading tree {branch_idx}: {e}")
        df = self.data_storage[branch_idx]
        return df.iloc[atom]

    def get_property_noNAN(
        self,
        matched_node_path: list = None,
        mol: Molecule = None,
        atom: int = None,
        property_name: str = None,
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_incremet_threshold: float = 0,
    ):
        """
        Get a property (property_name) of a atom from a matched DASH tree subgraph (node path) or from a molecule
        and atom index (don't use both (mol +atom) and matched_node_path at the same time). The last non NaN value
        in the hierarchy is returned.

        Parameters
        ----------
        matched_node_path : list
            List of node ids of the matched subgraph (node path) in the order of the traversal
        mol : Molecule
            RDKit molecule object in which the atom is located
        atom : int
            Atom index in the molecule of the atom to match
        property_name : str
            Name of the property to return

        Returns
        -------
        float
            The last non NaN value of the property in the hierarchy
        """
        if matched_node_path is None:
            if mol is None or atom is None:
                raise ValueError("Either matched_node_path or mol + atom must be provided")
            matched_node_path = self.match_new_atom(
                atom=atom,
                mol=mol,
                max_depth=max_depth,
                attention_threshold=attention_threshold,
                attention_increment_threshold=attention_incremet_threshold,
            )
        branch_idx = matched_node_path[0]
        if branch_idx not in self.data_storage:
            try:
                self.load_tree_and_data(
                    os.path.join(self.tree_folder_path, f"{branch_idx}.gz"),
                    os.path.join(self.tree_folder_path, f"{branch_idx}.h5"),
                )
            except Exception as e:
                print(f"Error loading tree {branch_idx}: {e}")
        df = self.data_storage[branch_idx]
        for atom in reversed(matched_node_path):
            if not np.isnan(df.iloc[atom][property_name]):
                return df.iloc[atom][property_name]
        Warning(f"No non NaN value found for {property_name} in hierarchy {matched_node_path}")
        return np.nan

    def _get_allAtoms_nodePaths(
        self,
        mol: Molecule,
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_incremet_threshold: float = 0,
    ):
        """
        Get all properties of all atoms in a molecule by matching them to DASH tree subgraphs

        Parameters
        ----------
        mol : Molecule
            RDKit molecule object for which the properties should be calculated

        Returns
        -------
        dict
            Dictionary containing the properties of all atoms
        """
        nodePathList = []
        neighbor_dict = init_neighbor_dict(mol, af=self.atom_feature_type)
        for atom in range(mol.GetNumAtoms()):
            try:
                node_path = self.match_new_atom(
                    atom,
                    mol,
                    max_depth=max_depth,
                    attention_threshold=attention_threshold,
                    attention_increment_threshold=attention_incremet_threshold,
                    neighbor_dict=neighbor_dict,
                )
                nodePathList.append(node_path)
            except Exception:
                nodePathList.append([])
        return nodePathList

    def get_molecules_partial_charges(
        self,
        mol: Molecule,
        norm_method: str = "std_weighted",
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_incremet_threshold: float = 0,
        verbose: bool = False,
        default_std_value: float = 0.1,
        chg_key: str = "result",
        chg_std_key: str = "stdDeviation",
        nodePathList=None,
    ):
        """
        Get the partial charges of all atoms in a molecule by matching them to DASH tree subgraphs and
        normalizing the charges of the matched atoms

        Parameters
        ----------
        mol : Molecule
            RDKit molecule object for which the partial charges should be calculated
        norm_method : str
            Method to normalize the partial charges, one of 'none', 'symmetric', 'std_weighted'
        max_depth : int
            Maximum depth of the tree to traverse
        attention_threshold : float
            Maximum cumulative attention value to traverse the tree
        attention_incremet_threshold : float
            Minimum attention increment to stop the traversal
        verbose : bool
            If True, print status messages
        default_std_value : float
            Default value to use for the standard deviation if it is 0
        chg_key : str
            Key of the partial charge in the DASH tree data
        chg_std_key : str
            Key of the partial charge standard deviation in the DASH tree data

        Returns
        -------
        dict
            Dictionary containing the partial charges, standard deviations and match depths of all atoms
        """
        return_list = []
        tree_raw_charges = []
        tree_charge_std = []
        tree_match_depth = []
        if nodePathList is None:
            nodePathList = self._get_allAtoms_nodePaths(
                mol,
                max_depth=max_depth,
                attention_threshold=attention_threshold,
                attention_incremet_threshold=attention_incremet_threshold,
            )
        for nodePath in nodePathList:
            try:
                chg_atom = self.get_property_noNAN(
                    matched_node_path=nodePath,
                    property_name=chg_key,
                )
                chg_std_atom = self.get_property_noNAN(
                    matched_node_path=nodePath,
                    property_name=chg_std_key,
                )
                tree_raw_charges.append(float(chg_atom))
                tmp_tree_std = float(chg_std_atom)
                tree_charge_std.append(tmp_tree_std if tmp_tree_std > 0 else default_std_value)
                tree_match_depth.append(len(nodePath) - 1)
            except Exception as e:
                raise e

        if verbose:
            print(f"Tree raw charges: {tree_raw_charges}")
        if norm_method == "none":
            return_list = tree_raw_charges
        elif norm_method == "symmetric":
            tot_charge_tree = sum(tree_raw_charges)
            tot_charge_mol = sum([x.GetFormalCharge() for x in mol.GetAtoms()])
            return_list = [x + (tot_charge_mol - tot_charge_tree) / len(tree_raw_charges) for x in tree_raw_charges]
        elif norm_method == "std_weighted":
            tot_charge_tree = sum(tree_raw_charges)
            tot_charge_mol = sum([x.GetFormalCharge() for x in mol.GetAtoms()])
            tot_std_tree = sum(tree_charge_std)
            return_list = [
                x + (tot_charge_mol - tot_charge_tree) * (y / tot_std_tree)
                for x, y in zip(tree_raw_charges, tree_charge_std)
            ]
        else:
            raise ValueError("norm_method must be one of 'none', 'symmetric', 'std_weighted'")
        if verbose:
            print(f"Tree normalized charges: {return_list}")
        return {
            "charges": return_list,
            "std": tree_charge_std,
            "match_depth": tree_match_depth,
        }

    def _get_attention_sorted_neighbours_bondVectors(self, mol, atom_idx, verbose=False):
        rdkit_neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if verbose:
            print(f"rdkit_neighbors: {[n.GetIdx() for n in rdkit_neighbors]}")
        node_path, atom_indices_in_subgraph = self.match_new_atom(atom=atom_idx, mol=mol, return_atom_indices=True)
        if verbose:
            print(f"node_path: {node_path}")
            print(f"atom_indices_in_subgraph: {atom_indices_in_subgraph}")
        neighbours = []
        if mol.GetAtomWithIdx(atom_idx).GetSymbol() == "H":
            neighbours.append(atom_indices_in_subgraph[0])
        else:
            for node_idx, atom_idx_in_subgraph in zip(node_path[1:], atom_indices_in_subgraph):
                tmp_node = self.tree_storage[node_path[0]][node_idx]
                if tmp_node[2] == 0:
                    neighbours.append(atom_idx_in_subgraph)
        if verbose:
            print(f"neighbours: {neighbours}")
        # add Hs
        for neighbor in rdkit_neighbors:
            if neighbor.GetSymbol() == "H":
                neighbours.append(neighbor.GetIdx())
        if verbose:
            print(f"neighbours with Hs: {neighbours}")
        bond_vectors = []
        for neighbor in neighbours:
            bond_vectors.append(
                mol.GetConformer().GetAtomPosition(atom_idx) - mol.GetConformer().GetAtomPosition(neighbor)
            )
        return bond_vectors

    def _project_dipole_to_bonds(self, bond_vectors, dipole, verbose=False):
        # 1. Try to orthogonalize bond vectors
        try:
            bond_vectors_orthogonal = [np.array(bond_vectors[0])]
            try:
                bond_vectors_orthogonal.append(
                    bond_vectors[1]
                    - np.dot(bond_vectors[1], bond_vectors_orthogonal[0])
                    / np.linalg.norm(bond_vectors_orthogonal[0])
                    * bond_vectors_orthogonal[0]
                )
                try:
                    bond_vectors_orthogonal.append(
                        bond_vectors[2]
                        - np.dot(bond_vectors[2], bond_vectors_orthogonal[0])
                        / np.linalg.norm(bond_vectors_orthogonal[0])
                        * bond_vectors_orthogonal[0]
                        - np.dot(bond_vectors[2], bond_vectors_orthogonal[1])
                        / np.linalg.norm(bond_vectors_orthogonal[1])
                        * bond_vectors_orthogonal[1]
                    )
                except IndexError:
                    pass
            except IndexError:
                pass
        except IndexError:
            return np.nan
        if verbose:
            print(bond_vectors_orthogonal)
        # 2. Project dipole on bond vectors and pad to 3 dimensions
        projected_dipoles = []
        for bond_vector in bond_vectors_orthogonal:
            projected_dipoles.append(np.dot(dipole, bond_vector) / np.linalg.norm(bond_vector))
        projected_dipoles = np.pad(
            projected_dipoles,
            pad_width=(0, 3 - len(projected_dipoles)),
            mode="constant",
        )
        # 3. Normalize projected dipoles to have same length as dipole
        vec_sum_projected_dipoles = np.linalg.norm(np.sum(projected_dipoles))
        scale_factor = np.linalg.norm(dipole) / vec_sum_projected_dipoles
        projected_dipoles = [x * scale_factor for x in projected_dipoles]
        return projected_dipoles

    def _get_dipole_from_bond_projection(self, mol, atom_idx, projected_dipoles, verbose=False):
        bond_vectors = self._get_attention_sorted_neighbours_bondVectors(mol, atom_idx)
        dipole = np.zeros(3)
        for projected_dipole, bond_vector in zip(projected_dipoles, bond_vectors):
            if verbose:
                print(dipole)
            dipole += projected_dipole * bond_vector
        return dipole

    def get_atomic_dipole_vector(
        self,
        mol,
        atom_idx,
        prop_keys=["dipole_bond_1", "dipole_bond_2", "dipole_bond_3"],
    ):
        node_path = self.match_new_atom(atom=atom_idx, mol=mol)
        x = self.get_property_noNAN(matched_node_path=node_path, property_name=prop_keys[0])
        y = self.get_property_noNAN(matched_node_path=node_path, property_name=prop_keys[1])
        z = self.get_property_noNAN(matched_node_path=node_path, property_name=prop_keys[2])
        dipole = self._get_dipole_from_bond_projection(mol, atom_idx, [x, y, z])
        return dipole

    def get_molecular_dipole_moment(
        self,
        mol: Molecule,
        inDebye: bool = True,
        chg_key: str = "result",
        chg_std_key: str = "std",
        sngl_cnf=True,
        nconfs=10,
        pruneRmsThresh=0.5,
        useExpTorsionAnglePrefs=False,
        useBasicKnowledge=False,
        add_atomic_dipoles=False,
        nodePathList=None,
    ):
        """
        Get the dipole moment of a molecule by matching all atoms to DASH tree subgraphs and
        summing the dipole moments of the matched atoms
        """
        chgs = self.get_molecules_partial_charges(
            mol=mol,
            norm_method="std_weighted",
            chg_key=chg_key,
            chg_std_key=chg_std_key,
            nodePathList=nodePathList,
        )["charges"]
        # check if mol has conformer, otherwise generate one
        if mol.GetNumConformers() == 0:
            # AllChem.EmbedMolecule(mol)
            cids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=nconfs,
                pruneRmsThresh=pruneRmsThresh,
                randomSeed=42,
                useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
                useBasicKnowledge=useBasicKnowledge,
                ETversion=2,
            )
            energies = []
            for cid in cids:
                AllChem.UFFOptimizeMolecule(mol, confId=cid)
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                energies.append(ff.CalcEnergy())
            min_idx = int(np.argmin(energies))
            # set conformer with lowest energy as default
            mol.GetConformer(min_idx).SetId(0)
            for i in range(1, len(cids)):
                mol.RemoveConformer(i)
        # center_of_mass = np.array(ComputeCentroid(mol.GetConformer()))
        # dipole_vecs = [np.array(mol.GetConformer().GetAtomPosition(i)) - center_of_mass for i in range(mol.GetNumAtoms())]
        # vec_sum = np.sum([chg * dipole_vec for chg, dipole_vec in zip(chgs, dipole_vecs)], axis=0)
        if sngl_cnf:
            vec_sum = np.sum(
                [chg * np.array(mol.GetConformer().GetAtomPosition(i)) for i, chg in enumerate(chgs)],
                axis=0,
            )
            if add_atomic_dipoles:
                for atom_idx in range(mol.GetNumAtoms()):
                    vec_sum += self.get_atomic_dipole_vector(mol, atom_idx)
            dipole = np.linalg.norm(vec_sum)
        else:
            dipole = np.zeros(mol.GetNumConformers())
            for conf_i, conf in enumerate(mol.GetConformers()):
                vec_sum = np.sum(
                    [chg * np.array(conf.GetAtomPosition(i)) for i, chg in enumerate(chgs)],
                    axis=0,
                )
                if add_atomic_dipoles:
                    for atom_idx in range(mol.GetNumAtoms()):
                        vec_sum += self.get_atomic_dipole_vector(mol, atom_idx)
                dipole[conf_i] = np.linalg.norm(vec_sum)
        if inDebye:
            dipole /= 0.393430307
        return dipole

    def get_molecular_polarizability(
        self,
        mol: Molecule,
        prop_key: str = "DFTD4:polarizability",
    ):
        """
        Get the polarizability of a molecule by matching all atoms to DASH tree subgraphs and
        summing the polarizabilities of the matched atoms
        """
        polarizabilities = []
        all_nodePaths = self._get_allAtoms_nodePaths(mol)
        for nodePath in all_nodePaths:
            polarizabilities.append(self.get_property_noNAN(matched_node_path=nodePath, property_name=prop_key))
        polarizability = np.sum(polarizabilities)
        return polarizability

    def get_molecules_feature_vector(
        self,
        mol: Molecule,
        properties_to_use: list = [
            "result",
            "dual",
            "mbis_dipole_strength",
        ],
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_incremet_threshold: float = 0,
        verbose=False,
    ):
        return_list = []
        if verbose:
            print(properties_to_use)
        all_nodePaths = self._get_allAtoms_nodePaths(
            mol=mol,
            max_depth=max_depth,
            attention_threshold=attention_threshold,
            attention_incremet_threshold=attention_incremet_threshold,
        )
        for nodePath in all_nodePaths:
            tmp_p = []
            try:
                for p in properties_to_use:
                    try:
                        prop = self.get_property_noNAN(matched_node_path=nodePath, property_name=p)
                        tmp_p.append(prop)
                    except Exception:
                        tmp_p.append(np.nan)
                return_list.append(tmp_p)
            except Exception:
                return_list.append([np.nan] * len(properties_to_use))
        return return_list

    def get_DASH_feature_dict_for_mol(
        self,
        mol: Molecule,
    ):
        return_dict = {}
        tmp_mbis = []
        tmp_dual = []
        tmp_conj = []
        num_atoms = mol.GetNumAtoms()
        for atom_idx in range(num_atoms):
            node_path = self.match_new_atom(atom=atom_idx, mol=mol)
            bidx = node_path[0]
            tmp_mbis.append(self.get_property_noNAN(matched_node_path=node_path, property_name="result"))
            tmp_dual.append(self.get_property_noNAN(matched_node_path=node_path, property_name="dual"))
            key = self.tree_storage[bidx][0][1]
            tmp_conj.append(self.atom_feature_type.afKey_2_afTuple[key][3])
        return_dict["DASH_max_abs_mbis"] = np.max(np.abs(tmp_mbis))
        return_dict["DASH_avg_abs_mbis"] = np.mean(np.abs(tmp_mbis))
        return_dict["DASH_>03_abs_mbis"] = np.sum([1 if x > 0.3 else 0 for x in tmp_mbis]) / num_atoms
        return_dict["DASH_dual_elec"] = np.sum([1 if x < -0.4 else 0 for x in tmp_dual]) / num_atoms
        return_dict["DASH_dual_nucl"] = np.sum([1 if x > 0.4 else 0 for x in tmp_dual]) / num_atoms
        return_dict["DASH_conj"] = np.sum([1 if x else 0 for x in tmp_conj]) / num_atoms
        return_dict["DASH_num_atoms"] = num_atoms
        return return_dict

    def explain_property(
        self,
        mol: Molecule,
        atom: int,
        property_name: str = "result",
        max_depth: int = 16,
        attention_threshold: float = 10,
        attention_incremet_threshold: float = 0,
        show_property_diff: bool = True,
        prop_unit: str = None,
        plot_size=(600, 400),
        useSVG=False,
    ):
        """
        Explain the value of a property of a atom in a molecule by matching it to a DASH tree subgraph and
        returning the property values of the matched atoms and the contribution of each atom added to the subgraph

        Parameters
        ----------
        mol : Molecule
            RDKit molecule object for which the property should be explained
        atom : int
            Atom index in the molecule of the atom to explain
        property_name : str
            Name of the property to explain (example: 'result', "mulliken", ...)
        max_depth : int
            Maximum depth of the tree to traverse
        attention_threshold : float
            Maximum cumulative attention value to traverse the tree
        attention_incremet_threshold : float
            Minimum attention increment to stop the traversal
        show_property_diff : bool
            If True, show the difference between the property values of the matched atoms

        Returns
        -------
        Image
            Image showing the molecule with the matched atoms highlighted and the contribution of each atom added to the subgraph
        """
        node_path, match_indices = self.match_new_atom(
            atom,
            mol,
            max_depth=max_depth,
            attention_threshold=attention_threshold,
            attention_increment_threshold=attention_incremet_threshold,
            return_atom_indices=True,
        )
        prop_per_node = [self.data_storage[node_path[0]].iloc[i][property_name] for i in node_path[1:]]
        if show_property_diff:
            prop_change_per_node = [prop_per_node[i + 1] - prop_per_node[i] for i in range(len(prop_per_node) - 1)]
            prop_change_per_node = [prop_per_node[0]] + prop_change_per_node
            text_per_atom = [f"{i}: {change:.2f}" for i, change in enumerate(prop_change_per_node)]
        else:
            text_per_atom = [f"{i}" for i in range(len(node_path))]
        if property_name == "result":
            property_name = "Partial charge"
            if prop_unit is None:
                prop_unit = "e"
        plot_title = f"Atom: 0 ({atom} in mol). \nSum of all contributions: {property_name} = {prop_per_node[-1]:.2f} {prop_unit}"
        return draw_mol_with_highlights_in_order(
            mol=mol,
            highlight_atoms=match_indices,
            highlight_bonds=[],
            text_per_atom=text_per_atom,
            plot_title=plot_title,
            plot_size=plot_size,
            useSVG=useSVG,
        )


def draw_mol_with_highlights_in_order(
    mol,
    highlight_atoms=[],
    highlight_bonds=[],
    text_per_atom=[],
    plot_title: str = None,
    plot_size=(600, 400),
    useSVG=False,
):
    color = (0, 0.6, 0.1)
    alphas = [1 - i / (len(highlight_atoms) + 4) for i in range(len(highlight_atoms) + 1)]
    athighlights = defaultdict(list)
    bthighlights = defaultdict(list)
    arads = {}
    brads = {}
    for i, atom in enumerate(highlight_atoms):
        athighlights[atom].append((color[0], color[1], color[2], alphas[i]))
        arads[atom] = 0.75
        if len(text_per_atom) < len(highlight_atoms):
            text_per_atom = [str(i) for i in highlight_atoms]
        mol.GetAtomWithIdx(atom).SetProp("atomNote", f"{text_per_atom[i]}")
    for i, bond in enumerate(highlight_bonds):
        bthighlights[bond].append((color[0], color[1], color[2], alphas[i + 1]))
        brads[bond] = 100
    if useSVG:
        d2d = rdMolDraw2D.MolDraw2DSVG(plot_size[0], plot_size[1])
    else:
        d2d = rdMolDraw2D.MolDraw2DCairo(plot_size[0], plot_size[1])
    dopts = d2d.drawOptions()
    dopts.scaleHighlightBondWidth = False
    # remove Hs
    mol_pic = Chem.RemoveHs(mol)
    AllChem.Compute2DCoords(mol_pic)
    if plot_title is not None:
        dopts.legendFontSize = 30
        d2d.DrawMoleculeWithHighlights(mol_pic, plot_title, dict(athighlights), dict(bthighlights), arads, brads)
    else:
        d2d.DrawMoleculeWithHighlights(mol_pic, "", dict(athighlights), dict(bthighlights), arads, brads)
    d2d.FinishDrawing()
    if useSVG:
        if not IPython:
            raise ImportError("IPython is not available, cannot use SVG")
        p = d2d.GetDrawingText().replace("svg:", "")
        img = IPython.display.SVG(data=p)
    else:
        bio = io.BytesIO(d2d.GetDrawingText())
        img = Image.open(bio)
    return img
