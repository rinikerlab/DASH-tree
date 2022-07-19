from serenityff.charge.gnn.utils.custom_data import (
    CustomData,
    GraphData,
    CustomGraphData,
)
import numpy as np
import pytest


@pytest.fixture
def NODE_FEATURES():
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


@pytest.fixture
def EDGE_INDEX():
    return np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)


@pytest.fixture
def EDGE_FEATURES():
    return np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])


@pytest.fixture
def NODE_POS_FEATURES():
    return np.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])


@pytest.fixture
def CUSTOM_DATA():
    return CustomData(smiles="abc", molecule_charge=-2)


@pytest.fixture
def GRAPH_DATA(NODE_FEATURES, EDGE_INDEX, EDGE_FEATURES, NODE_POS_FEATURES):
    return GraphData(
        node_features=NODE_FEATURES,
        edge_index=EDGE_INDEX,
        edge_features=EDGE_FEATURES,
        node_pos_features=NODE_POS_FEATURES,
    )


@pytest.fixture
def CUSTOM_GRAPH_DATA(NODE_FEATURES, EDGE_INDEX, EDGE_FEATURES, NODE_POS_FEATURES):
    return CustomGraphData(
        node_features=NODE_FEATURES,
        edge_index=EDGE_INDEX,
        edge_features=EDGE_FEATURES,
        node_pos_features=NODE_POS_FEATURES,
    )


def test_graph_data_basics(NODE_FEATURES, EDGE_INDEX, EDGE_FEATURES, NODE_POS_FEATURES):
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
        GraphData(NODE_FEATURES, EDGE_INDEX, 12)
    with pytest.raises(ValueError):
        GraphData(NODE_FEATURES, EDGE_INDEX, np.array([[1], [2], [3], [5]]))
    with pytest.raises(TypeError):
        GraphData(NODE_FEATURES, EDGE_INDEX, EDGE_FEATURES, 0)
    with pytest.raises(ValueError):
        GraphData(NODE_FEATURES, EDGE_INDEX, EDGE_FEATURES, np.array([[1], [2], [3], [5]]))
    GraphData(NODE_FEATURES, EDGE_INDEX, EDGE_FEATURES, NODE_POS_FEATURES)


def test_getters(
    GRAPH_DATA,
    CUSTOM_GRAPH_DATA,
    NODE_FEATURES,
    EDGE_INDEX,
    EDGE_FEATURES,
    NODE_POS_FEATURES,
):
    graph = GRAPH_DATA
    np.array_equal(graph.node_features, NODE_FEATURES)
    np.array_equal(graph.edge_index, EDGE_INDEX)
    np.array_equal(graph.edge_features, EDGE_FEATURES)
    np.array_equal(graph.node_pos_features, NODE_POS_FEATURES)
    assert graph.num_nodes == 5
    assert graph.num_node_features == 3
    assert graph.num_edges == 5
    assert graph.num_edge_features == 2

    graph = CUSTOM_GRAPH_DATA
    np.array_equal(graph.node_features, NODE_FEATURES)
    np.array_equal(graph.edge_index, EDGE_INDEX)
    np.array_equal(graph.edge_features, EDGE_FEATURES)
    np.array_equal(graph.node_pos_features, NODE_POS_FEATURES)
    assert graph.num_nodes == 5
    assert graph.num_node_features == 3
    assert graph.num_edges == 5
    assert graph.num_edge_features == 2


def test_to_pyg(
    CUSTOM_GRAPH_DATA,
    NODE_FEATURES,
    EDGE_INDEX,
    EDGE_FEATURES,
    NODE_POS_FEATURES,
):
    pyg_graph = CUSTOM_GRAPH_DATA.to_pyg_graph()
    assert isinstance(pyg_graph, CustomData)
    np.array_equal(pyg_graph.x.tolist(), NODE_FEATURES.tolist())
    np.array_equal(pyg_graph.edge_index.tolist(), EDGE_INDEX.tolist())
    np.array_equal(pyg_graph.edge_attr.tolist(), EDGE_FEATURES.tolist())
    np.array_equal(pyg_graph.pos.tolist(), NODE_POS_FEATURES.tolist())


def test_custom_data_attributes(CUSTOM_DATA):
    data = CUSTOM_DATA
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
    assert data.molecule_charge == 25
