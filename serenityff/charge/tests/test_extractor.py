from serenityff.charge.gnn import (
    Extractor,
    ChargeCorrectedNodeWiseAttentiveFP,
    get_graph_from_mol,
)
from serenityff.charge.utils import Molecule, command_to_shell_file
from serenityff.charge.utils.io import _split_sdf, _get_job_id
from typing import OrderedDict, Sequence
from rdkit import Chem
import os
import pytest
import torch


@pytest.fixture
def extractor() -> Extractor:
    return Extractor()


@pytest.fixture
def cwd() -> str:
    return os.path.dirname(__file__)


@pytest.fixture
def sdf_path() -> str:
    return "serenityff/charge/data/example.sdf"


@pytest.fixture
def mol(sdf_path) -> Molecule:
    return Chem.SDMolSupplier(sdf_path, removeHs=False)[0]


@pytest.fixture
def num_atoms(mol) -> int:
    return mol.GetNumAtoms()


@pytest.fixture
def num_bonds(mol) -> int:
    return mol.GetNumBonds()


@pytest.fixture
def formal_charge(mol) -> int:
    return Chem.GetFormalCharge(mol)


@pytest.fixture
def smiles(mol) -> str:
    return Chem.MolToSmiles(mol, canonical=False)


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
def args(sdf_path) -> Sequence[str]:
    return ["-s", sdf_path, "-m", "asdf.pt"]


def test_split_sdf(cwd, sdf_path) -> None:
    _split_sdf(
        sdf_file=sdf_path,
        directory=f"{cwd}/sdftest",
    )
    assert os.path.isdir(f"{cwd}/sdftest")
    for file in range(1, 4):
        assert os.path.isfile(f"{cwd}/sdftest/{file}.sdf")
        os.remove(f"{cwd}/sdftest/{file}.sdf")
    os.rmdir(f"{cwd}/sdftest/")
    return


def test_job_id(cwd) -> None:
    with open(f"{cwd}/id.txt", "w") as f:
        f.write("sdcep ab ein \n sdf <12345> saoeb <sd>")
    id = _get_job_id(file=f"{cwd}/id.txt")
    assert id == "12345"
    os.remove(f"{cwd}/id.txt")
    return


def test_graph_from_mol(mol, num_atoms, num_bonds, formal_charge, smiles) -> None:
    graph = get_graph_from_mol(mol=mol)
    assert graph.num_nodes == num_atoms
    assert graph.num_edges == num_bonds * 2
    assert graph.edge_attr.shape == torch.Size([84, 11])
    assert graph.molecule_charge.item() == formal_charge
    assert graph.smiles == smiles
    assert graph.x.shape[0] == num_atoms
    assert len(graph.y) == num_atoms
    return


def test_arg_parser(args, sdf_path) -> None:
    args = Extractor._parse_filenames(args)
    assert args.sdffile == sdf_path
    assert args.mlmodel == "asdf.pt"
    return


def test_script_writing(cwd) -> None:
    Extractor._write_worker(directory=cwd)
    Extractor._write_cleaner(directory=cwd)
    for file in [f"{cwd}/worker.sh", f"{cwd}/cleaner.sh"]:
        assert os.path.isfile(file)
        os.remove(file)
    return


def test_explainer_initialization(extractor, model) -> None:
    extractor._initialize_expaliner(model=model, epochs=1)
    return


def test_command_to_shell_file(cwd) -> None:
    command_to_shell_file("echo Hello World", f"{cwd}/test.sh")
    os.path.isfile(f"{cwd}/test.sh")
    with open(f"{cwd}/test.sh", "r") as f:
        text = f.read()
    assert text == "#!/bin/bash\n\necho Hello World"
    os.remove(f"{cwd}/test.sh")


# def test_run_extraction_local(extractor, args) -> None:
#     extractor.run_extraction_local(args)
#     return
