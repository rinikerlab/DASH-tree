"""
Unit and regression dev for the serenityff package.

Test featurizer in utils.py
"""

from typing import List, Sequence

import numpy as np
import pytest
from rdkit import Chem

from serenityff.charge.gnn import MolGraphConvFeaturizer
from serenityff.charge.gnn.utils.featurizer import (
    Featurizer,
    MolecularFeaturizer,
    _ChemicalFeaturesFactory,
    _construct_atom_feature,
    _construct_bond_feature,
    construct_hydrogen_bonding_info,
    get_atom_hydrogen_bonding_one_hot,
    get_atom_total_degree_one_hot,
    one_hot_encode,
)
from serenityff.charge.utils import Atom, Bond, Molecule


@pytest.fixture
def SMILES() -> str:
    return r"c1ccccc1C2CC/2=N\C(O)=C\F"


@pytest.fixture
def ALLOWABLE_SET() -> List[str]:
    return ["C", "H", "O"]


@pytest.fixture
def EMPTY_SET() -> List:
    return []


@pytest.fixture
def MOL(SMILES) -> Molecule:
    return Chem.AddHs(Chem.MolFromSmiles(SMILES))


@pytest.fixture
def ATOMS(MOL) -> Sequence[Atom]:
    return MOL.GetAtoms()


@pytest.fixture
def BONDS(MOL) -> Sequence[Bond]:
    return MOL.GetBonds()


def test_initialization() -> None:
    featurizer = Featurizer()
    featurizer = MolecularFeaturizer()
    featurizer = MolGraphConvFeaturizer()
    del featurizer
    return


def test_one_hot_encode(ATOMS, ALLOWABLE_SET) -> None:
    assert one_hot_encode(ATOMS[0].GetSymbol(), ALLOWABLE_SET) == [1.0, 0.0, 0.0]
    assert one_hot_encode(ATOMS[0].GetSymbol(), ALLOWABLE_SET, include_unknown_set=True) == [1.0, 0.0, 0.0, 0.0]
    assert one_hot_encode(ATOMS[9].GetSymbol(), ALLOWABLE_SET, include_unknown_set=False) == [0.0, 0.0, 0.0]
    return


def test_hbond_constructor(MOL) -> None:
    factory = _ChemicalFeaturesFactory.get_instance()
    import os

    from rdkit import RDConfig

    ownfactory = Chem.ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))

    assert [(feat.GetAtomIds()[0], feat.GetFamily()) for feat in factory.GetFeaturesForMol(MOL)] == [
        (feat.GetAtomIds()[0], feat.GetFamily()) for feat in ownfactory.GetFeaturesForMol(MOL)
    ]
    return


def test_H_bonding(MOL, ATOMS) -> None:
    hbond_infos = construct_hydrogen_bonding_info(MOL)
    assert get_atom_hydrogen_bonding_one_hot(ATOMS[11], hbond_infos) == [1.0, 1.0]
    return


def test_degree(ATOMS) -> None:
    assert np.where(get_atom_total_degree_one_hot(ATOMS[20])) == np.array([[1]])
    assert np.where(get_atom_total_degree_one_hot(ATOMS[11])) == np.array([[2]])
    assert np.where(get_atom_total_degree_one_hot(ATOMS[10])) == np.array([[3]])
    assert np.where(get_atom_total_degree_one_hot(ATOMS[7])) == np.array([[4]])
    return


def test_atom_feature(MOL, ATOMS, ALLOWABLE_SET) -> None:
    hbond_infos = construct_hydrogen_bonding_info(MOL)
    feat = _construct_atom_feature(
        ATOMS[9],
        h_bond_infos=hbond_infos,
        allowable_set=ALLOWABLE_SET,
        use_partial_charge=False,
    )
    np.testing.assert_array_equal(np.where(feat), np.array([[3, 6, 13]]))
    return


def test_bond_feat(BONDS) -> None:
    np.testing.assert_array_equal(np.where(_construct_bond_feature(BONDS[0])), np.array([[3, 4, 5, 6]]))
    np.testing.assert_array_equal(np.where(_construct_bond_feature(BONDS[6])), np.array([[0, 4, 6]]))
    np.testing.assert_array_equal(np.where(_construct_bond_feature(BONDS[8])), np.array([[1, 5, 9]]))
    np.testing.assert_array_equal(np.where(_construct_bond_feature(BONDS[11])), np.array([[1, 5, 8]]))
    return


def test_feature_vector_generation(SMILES, MOL, ALLOWABLE_SET, EMPTY_SET) -> None:
    featurizer = MolGraphConvFeaturizer(use_edges=True)

    with pytest.raises(AttributeError):
        featurizer._featurize(SMILES, allowable_set=ALLOWABLE_SET)

    graph = featurizer._featurize(datapoint=MOL, allowable_set=ALLOWABLE_SET).to_pyg_graph()
    empty_graph = featurizer._featurize(datapoint=MOL, allowable_set=EMPTY_SET).to_pyg_graph()

    for vec in empty_graph.x:
        assert vec[0].item() == 1

    assert len(graph.x[0]) == 18
    np.testing.assert_array_equal(np.where(graph.x[0]), np.array([[0, 6, 10, 14]]))
    np.testing.assert_array_equal(np.where(graph.x[7]), np.array([[0, 7, 15]]))
    np.testing.assert_array_equal(np.where(graph.x[9]), np.array([[3, 6, 13]]))

    np.testing.assert_array_equal(np.where(graph.edge_attr[22]), np.array([[1, 5, 8]]))
    np.testing.assert_array_equal(np.where(graph.edge_attr[23]), np.array([[1, 5, 8]]))
    return
