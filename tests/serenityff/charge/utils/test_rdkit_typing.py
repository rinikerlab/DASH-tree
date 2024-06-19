"""Test `serenityff.charge.utils.rdkit_typing`."""

import pytest
from rdkit import Chem

from serenityff.charge.utils.rdkit_typing import Atom, Bond, Molecule


@pytest.fixture()
def test_mol() -> Molecule:
    return Chem.AddHs(Chem.MolFromSmiles("C"))


def test_atom(test_mol: Molecule) -> None:
    """Test `Atom`."""
    assert isinstance(test_mol.GetAtomWithIdx(0), Atom)


def test_molecule(test_mol: Molecule) -> None:
    """Test `Molecule`."""
    assert isinstance(test_mol, Molecule)


def test_bond(test_mol: Molecule) -> None:
    """Test `Bond`."""
    assert isinstance(test_mol.GetBondWithIdx(0), Bond)
