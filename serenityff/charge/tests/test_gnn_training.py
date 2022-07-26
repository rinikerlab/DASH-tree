import pytest

# from numpy import array_equal
from rdkit import Chem
from torch import device
from torch.nn.functional import mse_loss
from torch.optim import Adam

from serenityff.charge.gnn import Trainer
from serenityff.charge.gnn.utils import ChargeCorrectedNodeWiseAttentiveFP, CustomData, get_graph_from_mol


@pytest.fixture
def sdf_path() -> str:
    return "serenityff/charge/data/example.sdf"


@pytest.fixture
def graphs_path() -> str:
    return "serenityff/charge/data/example_graphs.pt"


@pytest.fixture
def graph(sdf_path) -> CustomData:
    return get_graph_from_mol(Chem.SDMolSupplier(sdf_path, removeHs=False)[0])


@pytest.fixture
def model() -> ChargeCorrectedNodeWiseAttentiveFP:
    return ChargeCorrectedNodeWiseAttentiveFP(
        in_channels=25,
        hidden_channels=200,
        out_channels=1,
        edge_dim=11,
        num_layers=5,
        num_timesteps=2,
    )


@pytest.fixture
def optimizer(model):
    return Adam(model.parameters(), lr=0.001)


@pytest.fixture
def trainer():
    return Trainer()


@pytest.fixture
def lossfunction():
    return mse_loss


def test_init_and_forward_model(model, graph) -> None:
    model = model
    model.train()
    out = model(
        graph.x,
        graph.edge_index,
        graph.batch,
        graph.edge_attr,
        graph.molecule_charge,
    )
    assert len(out) == 41
    return


def test_initialize_trainer(trainer, model, optimizer, lossfunction, sdf_path, graphs_path) -> None:
    # test init
    assert trainer.device == device("cpu")

    # test setters
    trainer.model = model
    with pytest.raises(TypeError):
        trainer.model = "faulty"
    trainer.optimizer = optimizer
    with pytest.raises(TypeError):
        trainer.optimizer = "faulty"
    trainer.loss_function = lossfunction
    with pytest.raises(TypeError):
        trainer.loss_function = "faulty"
    trainer.device = "cuda"
    with pytest.raises(ValueError):
        trainer.device = "faulty type"
    with pytest.raises(TypeError):
        trainer.device = 2
    assert trainer.device == device("cuda")

    # test graph creation
    trainer.gen_graphs_from_sdf(sdf_path)
    assert len(trainer.data) == 3
    a = trainer.data
    trainer.load_graphs_from_pt(graphs_path)
    assert len(trainer.data) == 3
    b = trainer.data

    for x, y in zip(a, b):
        assert x.x == y.x
        assert x.batch == y.batch
        assert x.edge_index == y.edge_index
        assert x.edge_attr == y.edge_attr
        assert x.y == y.y
        assert x.smiles == y.smiles
        assert x.molecule_charge == y.molecule_charge
