from typing import List, Sequence

import numpy as np
import pytest
from rdkit import Chem

from serenityff.charge.gnn import MolGraphConvFeaturizer
from serenityff.charge.gnn.utils.custom_data import CustomData, CustomGraphData, GraphData
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
from serenityff.charge.gnn.utils.split_utils import get_split_numbers, split_data_Kfold, split_data_random
from serenityff.charge.utils import Atom, Bond, Molecule


@pytest.fixture
def data() -> Sequence[int]:
    return [*range(11)]


@pytest.fixture
def smiles() -> str:
    return r"c1ccccc1C2CC/2=N\C(O)=C\F"


@pytest.fixture
def allowable_set() -> List[str]:
    return ["C", "H", "O"]


@pytest.fixture
def empty_set() -> List:
    return []


@pytest.fixture
def mol(smiles) -> Molecule:
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


@pytest.fixture
def atoms(mol) -> Sequence[Atom]:
    return mol.GetAtoms()


@pytest.fixture
def bonds(mol) -> Sequence[Bond]:
    return mol.GetBonds()


@pytest.fixture
def edge_index() -> np.ndarray:
    return np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)


@pytest.fixture
def edge_features() -> np.ndarray:
    return np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])


@pytest.fixture
def node_pos_features() -> np.ndarray:
    return np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])


@pytest.fixture
def custom_data() -> CustomData:
    return CustomData(smiles="abc", molecule_charge=-2)


@pytest.fixture
def graph_data(node_features, edge_index, edge_features, node_pos_features) -> GraphData:
    return GraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_pos_features=node_pos_features,
    )


@pytest.fixture
def custom_graph_data(node_features, edge_index, edge_features, node_pos_features) -> CustomGraphData:
    return CustomGraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        node_pos_features=node_pos_features,
    )


@pytest.fixture
def node_features() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )


def test_get_split_numbers() -> None:
    assert [1, 0] == get_split_numbers(N=1, train_ratio=0.5)
    assert [1, 1] == get_split_numbers(N=2, train_ratio=0.5)
    assert [2, 1] == get_split_numbers(N=3, train_ratio=0.5)
    assert [50, 50] == get_split_numbers(N=100, train_ratio=0.5)
    assert [51, 50] == get_split_numbers(N=101, train_ratio=0.5)
    assert [51, 51] == get_split_numbers(N=102, train_ratio=0.5)
    return


def test_random_split(data) -> None:
    train, test = split_data_random(data_list=data, train_ratio=0.5)
    assert len(train) == 6
    assert len(test) == 5
    return


def test_kfold_split(data) -> None:
    train1, test1 = split_data_Kfold(data, n_splits=2, split=0)
    train2, test2 = split_data_Kfold(data, n_splits=2, split=1)
    assert not np.array_equal(train1, train2)
    assert not np.array_equal(test1, test2)
    return


def test_initialization() -> None:
    featurizer = Featurizer()
    featurizer = MolecularFeaturizer()
    featurizer = MolGraphConvFeaturizer()
    del featurizer
    return


def test_one_hot_encode(atoms, allowable_set) -> None:
    assert one_hot_encode(atoms[0].GetSymbol(), allowable_set) == [1.0, 0.0, 0.0]
    assert one_hot_encode(atoms[0].GetSymbol(), allowable_set, include_unknown_set=True) == [1.0, 0.0, 0.0, 0.0]
    assert one_hot_encode(atoms[9].GetSymbol(), allowable_set, include_unknown_set=False) == [0.0, 0.0, 0.0]
    return


def test_hbond_constructor(mol) -> None:
    factory = _ChemicalFeaturesFactory.get_instance()
    import os

    from rdkit import RDConfig

    ownfactory = Chem.ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))

    assert [(feat.GetAtomIds()[0], feat.GetFamily()) for feat in factory.GetFeaturesForMol(mol)] == [
        (feat.GetAtomIds()[0], feat.GetFamily()) for feat in ownfactory.GetFeaturesForMol(mol)
    ]
    return


def test_H_bonding(mol, atoms) -> None:
    hbond_infos = construct_hydrogen_bonding_info(mol)
    assert get_atom_hydrogen_bonding_one_hot(atoms[11], hbond_infos) == [1.0, 1.0]
    return


def test_degree(atoms) -> None:
    assert np.where(get_atom_total_degree_one_hot(atoms[20])) == np.array([[1]])
    assert np.where(get_atom_total_degree_one_hot(atoms[11])) == np.array([[2]])
    assert np.where(get_atom_total_degree_one_hot(atoms[10])) == np.array([[3]])
    assert np.where(get_atom_total_degree_one_hot(atoms[7])) == np.array([[4]])
    return


def test_atom_feature(mol, atoms, allowable_set) -> None:
    hbond_infos = construct_hydrogen_bonding_info(mol)
    feat = _construct_atom_feature(
        atoms[9],
        h_bond_infos=hbond_infos,
        allowable_set=allowable_set,
        use_partial_charge=False,
    )
    np.testing.assert_array_equal(np.where(feat), np.array([[3, 6, 13]]))
    return


def test_bond_feat(bonds) -> None:
    np.testing.assert_array_equal(np.where(_construct_bond_feature(bonds[0])), np.array([[3, 4, 5, 6]]))
    np.testing.assert_array_equal(np.where(_construct_bond_feature(bonds[6])), np.array([[0, 4, 6]]))
    np.testing.assert_array_equal(np.where(_construct_bond_feature(bonds[8])), np.array([[1, 5, 9]]))
    np.testing.assert_array_equal(np.where(_construct_bond_feature(bonds[11])), np.array([[1, 5, 8]]))
    return


def test_feature_vector_generation(smiles, mol, allowable_set, empty_set) -> None:
    featurizer = MolGraphConvFeaturizer(use_edges=True)

    with pytest.raises(AttributeError):
        featurizer._featurize(smiles, allowable_set=allowable_set)

    graph = featurizer._featurize(datapoint=mol, allowable_set=allowable_set).to_pyg_graph()
    empty_graph = featurizer._featurize(datapoint=mol, allowable_set=empty_set).to_pyg_graph()

    for vec in empty_graph.x:
        assert vec[0].item() == 1

    assert len(graph.x[0]) == 18
    np.testing.assert_array_equal(np.where(graph.x[0]), np.array([[0, 6, 10, 14]]))
    np.testing.assert_array_equal(np.where(graph.x[7]), np.array([[0, 7, 15]]))
    np.testing.assert_array_equal(np.where(graph.x[9]), np.array([[3, 6, 13]]))

    np.testing.assert_array_equal(np.where(graph.edge_attr[22]), np.array([[1, 5, 8]]))
    np.testing.assert_array_equal(np.where(graph.edge_attr[23]), np.array([[1, 5, 8]]))
    return


def test_graph_data_basics(node_features, edge_index, edge_features, node_pos_features) -> None:
    # Trigger all __init__() failures.
    with pytest.raises(TypeError):
        GraphData(11, 2)
    with pytest.raises(TypeError):
        GraphData(np.array([]), 2)
    with pytest.raises(TypeError):
        GraphData(np.array([]), np.array([2.2]))
    with pytest.raises(ValueError):
        GraphData(np.array([]), np.array([2, 1, 2]))
    with pytest.raises(ValueError):
        GraphData(np.array([]), np.array([2, 2]))
    with pytest.raises(TypeError):
        GraphData(node_features, edge_index, 12)
    with pytest.raises(ValueError):
        GraphData(node_features, edge_index, np.array([[1], [2], [3], [5]]))
    with pytest.raises(TypeError):
        GraphData(node_features, edge_index, edge_features, 0)
    with pytest.raises(ValueError):
        GraphData(node_features, edge_index, edge_features, np.array([[1], [2], [3], [5]]))
    GraphData(node_features, edge_index, edge_features, node_pos_features)
    return


def test_getters(
    graph_data,
    custom_graph_data,
    node_features,
    edge_index,
    edge_features,
    node_pos_features,
) -> None:
    graph = graph_data
    np.array_equal(graph.node_features, node_features)
    np.array_equal(graph.edge_index, edge_index)
    np.array_equal(graph.edge_features, edge_features)
    np.array_equal(graph.node_pos_features, node_pos_features)
    assert graph.num_nodes == 5
    assert graph.num_node_features == 3
    assert graph.num_edges == 5
    assert graph.num_edge_features == 2

    graph = custom_graph_data
    np.array_equal(graph.node_features, node_features)
    np.array_equal(graph.edge_index, edge_index)
    np.array_equal(graph.edge_features, edge_features)
    np.array_equal(graph.node_pos_features, node_pos_features)
    assert graph.num_nodes == 5
    assert graph.num_node_features == 3
    assert graph.num_edges == 5
    assert graph.num_edge_features == 2
    return


def test_to_pyg(
    custom_graph_data,
    node_features,
    edge_index,
    edge_features,
    node_pos_features,
) -> None:
    pyg_graph = custom_graph_data.to_pyg_graph()
    assert isinstance(pyg_graph, CustomData)
    np.array_equal(pyg_graph.x.tolist(), node_features.tolist())
    np.array_equal(pyg_graph.edge_index.tolist(), edge_index.tolist())
    np.array_equal(pyg_graph.edge_attr.tolist(), edge_features.tolist())
    np.array_equal(pyg_graph.pos.tolist(), node_pos_features.tolist())
    return


def test_custom_data_attributes(custom_data) -> None:
    data = custom_data
    assert data.smiles == "abc"
    assert data.molecule_charge == -2
    with pytest.raises(TypeError):
        data.smiles = 3
    with pytest.raises(TypeError):
        data.molecule_charge = "a"
    with pytest.raises(TypeError):
        data.molecule_charge = [2, 3]
    with pytest.raises(TypeError):
        data.molecule_charge = 2.1
    data.smiles = "C@@@"
    assert data.smiles == "C@@@"
    data.molecule_charge = 25
    data.molecule_charge = 25.0000
    assert data.molecule_charge.item() == 25
    return


def test_base_featurizer():
    featurizer = Featurizer()
    datapoints = [1]
    assert str(featurizer) == "Featurizer"
    assert repr(featurizer) == "Featurizer[]"
    with pytest.raises(NotImplementedError):
        featurizer._featurize(datapoints)
    features = featurizer.featurize(datapoints, log_every_n=1)
    assert np.array_equal(features, np.asarray([np.array([])]))


def test_molecular_featurizer(mol, smiles):
    featurizer = MolecularFeaturizer()
    featurizer.featurize([mol])
    featurizer.featurize(mol)
    featurizer.featurize([smiles])
    featurizer.featurize(smiles)
