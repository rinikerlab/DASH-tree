from serenityff.charge.gnn import Extractor
from rdkit import Chem
import os
import pytest


@pytest.fixture
def extractor():
    return Extractor()


@pytest.fixture
def cwd():
    return os.path.dirname(__file__)


@pytest.fixture
def sdf_path():
    return "serenityff/charge/data/example.sdf"


@pytest.fixture
def mol(sdf_path):
    return Chem.SDMolSupplier(sdf_path, removeHs=False)[0]


@pytest.fixture
def num_atoms(mol):
    return mol.GetNumAtoms()


@pytest.fixture
def num_bonds(mol):
    return mol.GetNumBonds()


@pytest.fixture
def formal_charge(mol):
    return Chem.GetFormalCharge(mol)


@pytest.fixture
def smiles(mol):
    return Chem.MolToSmiles(mol, canonical=False)


def test_split_sdf(cwd, sdf_path):
    Extractor._split_sdf(
        sdf_file=sdf_path,
        directory=f"{cwd}/sdftest",
    )
    assert os.path.isdir(f"{cwd}/sdftest")
    for file in range(1, 4):
        assert os.path.isfile(f"{cwd}/sdftest/{file}.sdf")
        os.remove(f"{cwd}/sdftest/{file}.sdf")
    os.rmdir(f"{cwd}/sdftest/")


def test_job_id(cwd):
    with open(f"{cwd}/id.txt", "w") as f:
        f.write("sdcep ab ein \n sdf <12345> saoeb <sd>")
    id = Extractor._get_job_id(file=f"{cwd}/id.txt")
    assert id == "12345"
    os.remove(f"{cwd}/id.txt")


def test_graph_from_mol(mol, num_atoms, num_bonds, formal_charge, smiles):
    graph = Extractor._get_graph_from_mol(mol=mol)
    assert graph.num_nodes == num_atoms
    assert graph.num_edges == num_bonds * 2
    assert graph.molecule_charge.item() == formal_charge
    assert graph.smiles == smiles
    assert graph.x.shape[0] == num_atoms
    assert len(graph.y) == num_atoms


def test_arg_parser(sdf_path):
    args = Extractor._parse_filenames(["-s", sdf_path, "-m", "asdf.pt"])
    assert args.sdffile == sdf_path
    assert args.mlmodel == "asdf.pt"


def test_script_writing(cwd):
    Extractor._write_worker(directory=cwd)
    Extractor._write_cleaner(directory=cwd)
    for file in [f"{cwd}/worker.sh", f"{cwd}/cleaner.sh"]:
        assert os.path.isfile(file)
        os.remove(file)
