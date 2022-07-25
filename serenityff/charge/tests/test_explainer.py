from serenityff.charge.gnn.attention_extraction import Explainer
from serenityff.charge.gnn.utils.model import ChargeCorrectedNodeWiseAttentiveFP
from serenityff.charge.gnn.utils import CustomData
from serenityff.charge.gnn import get_graph_from_mol
from torch_geometric.nn import GNNExplainer
from typing import OrderedDict
from rdkit import Chem
import numpy as np
import pytest
import torch


@pytest.fixture
def statedict() -> OrderedDict:
    return torch.load("serenityff/charge/data/example_state_dict.pt")


@pytest.fixture
def model(statedict) -> ChargeCorrectedNodeWiseAttentiveFP:
    m = ChargeCorrectedNodeWiseAttentiveFP(
        in_channels=25,
        hidden_channels=200,
        out_channels=1,
        edge_dim=11,
        num_layers=5,
        num_timesteps=2,
    )
    m.load_state_dict(statedict)
    return m


@pytest.fixture
def explainer(model) -> Explainer:
    return Explainer(model=model, epochs=1, verbose=True)


@pytest.fixture
def graph() -> CustomData:
    return get_graph_from_mol(Chem.SDMolSupplier("serenityff/charge/data/example.sdf", removeHs=False)[0])


def test_getter_setter(explainer) -> None:
    with pytest.raises(TypeError):
        explainer.gnn_explainer = "asdf"
    assert isinstance(explainer.gnn_explainer, GNNExplainer)
    assert explainer.gnnverbose
    explainer.gnnverbose = False
    assert not explainer.gnnverbose
    return


def test_load(model, statedict) -> None:
    np.array_equal(model.state_dict(), statedict)
    return


def test_explain_atom(explainer, graph) -> None:
    print(graph.x.shape, graph.edge_index.shape, graph.edge_attr.shape)
    explainer.gnn_explainer.explain_node(
        node_idx=0,
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.batch,
        molecule_charge=graph.molecule_charge,
    )
    an, ae = explainer._explain(
        node_idx=0,
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.batch,
        molecule_charge=graph.molecule_charge,
    )
    bn, be = explainer._explain_atom(node_idx=0, graph=graph)
    cn, ce = explainer.explain_molecule(graph=graph)
    np.array_equal(an, bn)
    np.array_equal(ae, be)
    np.array_equal(an, cn[0])
    np.array_equal(ae, ce[0])
    explainer.gnn_explainer.explain_node(
        0,
        graph.x,
        graph.edge_index,
        edge_attr=graph.edge_attr,
        batch=graph.batch,
        molecule_charge=graph.molecule_charge,
    )
    return
