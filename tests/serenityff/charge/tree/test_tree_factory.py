from collections.abc import Callable, Sequence

import numpy as np
import pytest
from rdkit import Chem

from serenityff.charge.tree.dash_tree import DASHTree
from serenityff.charge.tree.tree_factory import Forest
from tests._testfiles import (
    TEST_C6_JSON,
    TEST_CHARGES_JSON,
    TEST_DIPOLE_JSON,
    TEST_DUAL_DESCRIPTOR_JSON,
)
from tests._utils import are_in_CI, read_json


@pytest.fixture()
def test_mol() -> Chem.Mol:
    return Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))


@pytest.fixture()
def charge_solutions() -> dict:
    return read_json(TEST_CHARGES_JSON)


@pytest.mark.parametrize(
    "constructor, value_column, std_column",
    [
        (Forest.get_MBIS_DASH_tree, "result", "std"),
        (Forest.get_AM1BCC_DASH_tree, "AM1BCC", "AM1BCC_std"),
        (Forest.get_RESP1_DASH_tree, "resp1", "std"),
        (Forest.get_RESP2_DASH_tree, "resp2", "std"),
        (Forest.get_mulliken_DASH_tree, "mulliken", "std"),
        (Forest.get_charges_DASH_tree, "result", "std"),
        (Forest.get_dual_descriptor_DASH_tree, "dual", "std"),
        (Forest.get_dipole_DASH_tree, "mbis_dipole_strength", "std"),
        (Forest.get_C6_DASH_tree, "DFTD4:C6", "DFTD4:C6_std"),
        (
            Forest.get_polarizability_DASH_tree,
            "DFTD4:polarizability",
            "DFTD4:polarizability_std",
        ),
        (Forest.get_full_props_DASH_tree, "result", "std"),
    ],
)
def test_DASH_tree_columns(constructor: Callable, value_column: str, std_column: str) -> None:
    """Test that the correct columns are assigned upon construction.

    Args:
        constructor (Callable): constructor to test
        value_column (str): value column that should be set
        std_column (str): std column that should be set
    """
    tree: DASHTree = constructor(preload=False)
    assert tree.default_value_column == value_column
    assert tree.default_std_column == std_column


def test_mbis_charges(
    charge_solutions: dict,
    test_mol: Chem.Mol,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test that mbis charges are correct with default tree.

    Args:
        charge_solutions (dict): solutions dict from json file
        test_mol (Chem.Mol): test molecule
        caplog: (pytest.CaptureFixture): pytest magic
    """
    tree = Forest.get_MBIS_DASH_tree(preload=True)
    charge, std, match_depth = tree.get_molecules_partial_charges(test_mol).values()
    solution = charge_solutions["mbis"]
    np.testing.assert_array_almost_equal(charge, solution["charges"], decimal=3)
    np.testing.assert_array_almost_equal(std, solution["std"], decimal=3)
    np.testing.assert_array_almost_equal(match_depth, solution["match_depth"])
    assert (
        "The DASH Tree is missing additional data and will install that. This Can take a few minutes..."
        not in capsys.readouterr().out
    )


@pytest.mark.slow()
@pytest.mark.skipif(condition=are_in_CI(), reason="Takes too long on CI")
@pytest.mark.parametrize(
    "constructor, charge_solution_key",
    [
        (Forest.get_MBIS_DASH_tree, "mbis"),
        (Forest.get_AM1BCC_DASH_tree, "am1bcc"),
        (Forest.get_RESP1_DASH_tree, "resp1"),
        (Forest.get_RESP2_DASH_tree, "resp2"),
        (Forest.get_mulliken_DASH_tree, "mulliken"),
        (Forest.get_charges_DASH_tree, "mbis"),
    ],
)
def test_different_charges(
    constructor: Callable,
    charge_solution_key: Sequence[float],
    charge_solutions: dict,
    test_mol: Chem.Mol,
) -> None:
    """Test different charge trees.

    Args:
        tree (Callable): charge tree constructor to test
        charge_solution_key (Sequence[float]): key to the solution json file
        charge_solutions (dict): solution dict from json file
        test_mol (Chem.Mol): test molecule
    """
    tree: DASHTree = constructor(preload=True)
    charge, std, match_depth = tree.get_molecules_partial_charges(test_mol).values()
    solution = charge_solutions[charge_solution_key]
    np.testing.assert_array_almost_equal(charge, solution["charges"], decimal=3)
    np.testing.assert_array_almost_equal(std, solution["std"], decimal=3)
    np.testing.assert_array_almost_equal(match_depth, solution["match_depth"])


@pytest.mark.slow()
@pytest.mark.skipif(condition=are_in_CI(), reason="Takes too long on CI")
@pytest.mark.parametrize(
    "charge_column, json_key, std_column",
    [
        ("result", "mbis", "std"),
        ("AM1BCC", "am1bcc", "AM1BCC_std"),
        ("resp1", "resp1", "std"),
        ("resp2", "resp2", "std"),
        ("mulliken", "mulliken", "std"),
    ],
)
def test_full_charges_tree(
    charge_column: str,
    json_key: str,
    std_column: str,
    charge_solutions: dict,
    test_mol: Chem.Mol,
) -> None:
    """Test that the full charges tree has all charge values.

    Args:
        charge_column (str): value column to test
        json_key (str): json key that has the values for this column
        std_column (str): std column to test
        charge_solutions (dict): dict from the solutions json file
        test_mol (Chem.Mol): test molecule
    """
    tree = Forest.get_charges_DASH_tree(preload=True)
    charge, std, match_depth = tree.get_molecules_partial_charges(
        mol=test_mol, chg_key=charge_column, chg_std_key=std_column
    ).values()
    solution = charge_solutions[json_key]
    np.testing.assert_array_almost_equal(charge, solution["charges"], decimal=3)
    np.testing.assert_array_almost_equal(std, solution["std"], decimal=3)
    np.testing.assert_array_almost_equal(match_depth, solution["match_depth"])


@pytest.mark.slow()
@pytest.mark.skipif(condition=are_in_CI(), reason="Takes too long on CI")
def test_dipole_tree(test_mol: Chem.Mol) -> None:
    """Test the dipole tree.

    Args:
        test_mol (Chem.Mol): test molecule
    """
    params = Chem.rdDistGeom.EmbedParameters()
    params.randomSeed = 0xF00D
    Chem.rdDistGeom.EmbedMolecule(test_mol, params)
    tree = Forest.get_dipole_DASH_tree(preload=True)
    solution = read_json(TEST_DIPOLE_JSON)
    for atom in test_mol.GetAtoms():
        atom_index = atom.GetIdx()
        dipole = tree.get_atomic_dipole_vector(mol=test_mol, atom_idx=atom_index)
        np.testing.assert_array_almost_equal(dipole, solution[str(atom_index)])


@pytest.mark.slow()
@pytest.mark.skipif(condition=are_in_CI(), reason="Takes too long on CI")
def test_dual_tree(test_mol: Chem.Mol) -> None:
    """Test the dual descriptor tree.

    Args:
        test_mol (Chem.Mol): test molecule
    """
    tree = Forest.get_dual_descriptor_DASH_tree(preload=True)
    solution = read_json(TEST_DUAL_DESCRIPTOR_JSON)
    for atom in test_mol.GetAtoms():
        atom_index = atom.GetIdx()
        dual_descriptor = tree.get_property_noNAN(mol=test_mol, atom=atom_index)
        np.testing.assert_almost_equal(dual_descriptor, solution[str(atom_index)])


@pytest.mark.slow()
@pytest.mark.skipif(condition=are_in_CI(), reason="Takes too long on CI")
def test_c6_tree(test_mol: Chem.Mol) -> None:
    """Test the DFTD4 C6 tree.

    Args:
        test_mol (Chem.Mol): test molecule
    """
    tree = Forest.get_C6_DASH_tree(preload=True)
    solution = read_json(TEST_C6_JSON)
    for atom in test_mol.GetAtoms():
        atom_index = atom.GetIdx()
        c6 = tree.get_property_noNAN(mol=test_mol, atom=atom_index)
        np.testing.assert_almost_equal(c6, solution[str(atom_index)], decimal=3)


@pytest.mark.slow()
@pytest.mark.skipif(condition=are_in_CI(), reason="Takes too long on CI")
def test_polarizability_tree(test_mol: Chem.Mol) -> None:
    """Test DFTD4 polarizability tree.

    Args:
        test_mol (Chem.Mol): test molecule
    """
    tree = Forest.get_polarizability_DASH_tree(preload=True)
    polarizability = tree.get_molecular_polarizability(mol=test_mol)
    np.testing.assert_almost_equal(polarizability, 72.9449, decimal=3)
