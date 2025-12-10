# Copyright (C) 2022-2025 ETH Zurich, Niels Maeder and other DASH contributors.

from typing import List, Optional, Sequence

import numpy as np
import torch as pt
from rdkit import Chem

from serenityff.charge.gnn.utils import CustomData, MolGraphConvFeaturizer
from serenityff.charge.utils import Molecule


def mols_from_sdf(sdf_file: str, removeHs: Optional[bool] = False) -> Sequence[Molecule]:
    """Return a sequence of RDKit molecules read from an .sdf file.

    :param sdf_file: Path to the .sdf file.
    :param removeHs: Whether to remove hydrogens. Defaults to False.
    :return: A sequence of RDKit molecule objects.
    """
    return Chem.SDMolSupplier(sdf_file, removeHs=removeHs)


def get_mol_prop_as_np_array(prop_name: Optional[str], mol: Chem.Mol, dtype: type = float) -> np.ndarray:
    """Get atomic properties from an RDKit molecule object as an array.

    The property is expected to be a string of '|' separated numerical
    values, one for each atom in the molecule.

    :param prop_name: The name of the property to retrieve from the molecule.
    :param mol: The RDKit molecule object.
    :return: The atomic properties converted to a NumPy array.
    :raises ValueError: If ``prop_name`` is None or if the property is not found in the molecule.
    :raises TypeError: If any of the parsed property values are NaN or not convertible to float.
    """
    if prop_name is None:
        raise ValueError("Property name can not be None.")
    if not mol.HasProp(prop_name):
        raise ValueError(f"Property {prop_name} not found in molecule.")  # noqa E713
    array = np.fromstring(mol.GetProp(prop_name), sep="|", dtype=dtype)
    if np.isnan(array).any():
        raise TypeError(f"Nan found in {prop_name}.")
    return array


def get_mol_prop_as_pt_tensor(prop_name: Optional[str], mol: Chem.Mol) -> pt.Tensor:
    """Get atomic properties from an RDKit molecule object as a tensor.

    The property is expected to be a string of '|' separated numerical
    values, one for each atom in the molecule.

    :param prop_name: The name of the property to retrieve from the molecule.
    :param mol: The RDKit molecule object.
    :return: The atomic properties converted to a PyTorch tensor.
    :raises ValueError: If ``prop_name`` is None or if the property is not found in the molecule.
    :raises TypeError: If any of the parsed property values are NaN or not convertible to float.
    """
    return pt.from_numpy(get_mol_prop_as_np_array(prop_name=prop_name, mol=mol, dtype=np.float32))


def get_graph_from_mol(
    mol: Molecule,
    index: int,
    sdf_property_name: Optional[str] = "MBIScharge",
    allowable_set: Optional[List[str]] = [
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "H",
    ],
    no_y: Optional[bool] = False,
) -> Optional[CustomData]:
    """Create a PyTorch Geometric graph from an RDKit molecule.

    Returns ``None`` if the specified property is not found or contains NaN.

    The graph contains the following features:

        **Node features**
            - Atom type (as specified in the `allowable_set`)
            - Formal charge
            - Hybridization
            - H acceptor/donor
            - Aromaticity
            - Degree

        **Edge features**
            - Bond type
            - Is in ring
            - Is conjugated
            - Stereo information

    :param mol: The RDKit molecule.
    :param sdf_property_name: Name of the property in the SDF file to be used for training.
    :param allowable_set: List of atoms to include in the feature vector. Defaults to
        ``["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"]``.
    :return: A PyTorch Geometric ``Data`` object with an additional ``.smiles`` attribute,
        or ``None`` if the property is invalid.
    """
    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if no_y:
        graph.y = pt.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=pt.float,
        )
    else:
        try:
            graph.y = get_mol_prop_as_pt_tensor(sdf_property_name, mol)
        except TypeError as exc:
            print(exc)
            return None

    graph.batch = pt.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph
