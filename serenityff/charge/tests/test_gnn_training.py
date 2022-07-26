import pytest
from rdkit import Chem

from serenityff.charge.gnn import ChargeCorrectedNodeWiseAttentiveFP
from serenityff.charge.gnn.utils import CustomData, get_graph_from_mol


@pytest.fixture
def graph() -> CustomData:
    return get_graph_from_mol(Chem.SDMolSupplier("serenityff/charge/data/example.sdf", removeHs=False)[0])


@pytest.fixture
def model(statedict) -> ChargeCorrectedNodeWiseAttentiveFP:
    return ChargeCorrectedNodeWiseAttentiveFP(
        in_channels=25,
        hidden_channels=200,
        out_channels=1,
        edge_dim=11,
        num_layers=5,
        num_timesteps=2,
    )
