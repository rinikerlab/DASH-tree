# Copyright (C) 2025 ETH Zurich, Niels Maeder and other DASH contributors.

import numpy as np
import pytest
import torch as pt
from rdkit import Chem

from serenityff.charge.gnn.utils.rdkit_helper import (
    get_mol_prop_as_np_array,
    get_mol_prop_as_pt_tensor,
)


@pytest.fixture
def sample_mol_with_prop():
    """Fixture for a sample RDKit molecule with a valid property."""
    mol = Chem.MolFromSmiles("CCO")  # Ethanol
    mol.SetProp("test_prop", "1.0|2.5|-3.0")
    return mol


@pytest.fixture
def sample_mol_with_nan_prop():
    """Fixture for a sample RDKit molecule with a property containing NaN."""
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("test_prop_nan", "1.0|nan|3.0")
    return mol


@pytest.fixture
def sample_mol_missing_prop():
    """Fixture for a sample RDKit molecule without the desired property."""
    mol = Chem.MolFromSmiles("CCO")
    return mol


@pytest.mark.max
def test_get_mol_prop_as_pt_tensor_success(sample_mol_with_prop):
    """Test successful retrieval of property as a tensor."""
    expected = pt.tensor([1.0, 2.5, -3.0], dtype=pt.float)
    result = get_mol_prop_as_pt_tensor("test_prop", sample_mol_with_prop)
    assert isinstance(result, pt.Tensor)
    assert pt.equal(result, expected)


@pytest.mark.max
def test_get_mol_prop_as_pt_tensor_raises_value_error_on_none_prop(
    sample_mol_missing_prop,
):
    """Test ValueError is raised when prop_name is None."""
    with pytest.raises(ValueError, match="Property name can not be None"):
        get_mol_prop_as_pt_tensor(None, sample_mol_missing_prop)


@pytest.mark.max
def test_get_mol_prop_as_pt_tensor_raises_value_error_on_missing_prop(
    sample_mol_missing_prop,
):
    """Test ValueError is raised when the property is not found."""
    with pytest.raises(ValueError, match="Property missing_prop not found"):
        get_mol_prop_as_pt_tensor("missing_prop", sample_mol_missing_prop)


@pytest.mark.max
def test_get_mol_prop_as_pt_tensor_raises_type_error_on_nan(
    sample_mol_with_nan_prop,
):
    """Test TypeError is raised when NaN is in the property string."""
    with pytest.raises(TypeError, match="Nan found in test_prop_nan"):
        get_mol_prop_as_pt_tensor("test_prop_nan", sample_mol_with_nan_prop)


@pytest.mark.max
def test_get_mol_prop_as_np_array_success(sample_mol_with_prop):
    """Test successful retrieval of property as a numpy array."""
    expected = np.array([1.0, 2.5, -3.0])
    result = get_mol_prop_as_np_array("test_prop", sample_mol_with_prop)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.max
def test_get_mol_prop_as_np_array_raises_value_error_on_none_prop(
    sample_mol_missing_prop,
):
    """Test ValueError is raised when prop_name is None."""
    with pytest.raises(ValueError, match="Property name can not be None"):
        get_mol_prop_as_np_array(None, sample_mol_missing_prop)


@pytest.mark.max
def test_get_mol_prop_as_np_array_raises_value_error_on_missing_prop(
    sample_mol_missing_prop,
):
    """Test ValueError is raised when the property is not found."""
    with pytest.raises(ValueError, match="Property missing_prop not found"):
        get_mol_prop_as_np_array("missing_prop", sample_mol_missing_prop)


@pytest.mark.max
def test_get_mol_prop_as_np_array_raises_type_error_on_nan(
    sample_mol_with_nan_prop,
):
    """Test TypeError is raised when NaN is in the property string."""
    with pytest.raises(TypeError, match="Nan found in test_prop_nan"):
        get_mol_prop_as_np_array("test_prop_nan", sample_mol_with_nan_prop)
