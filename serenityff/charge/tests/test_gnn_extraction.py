import os
from shutil import rmtree
from typing import OrderedDict, Sequence

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from serenityff.charge.gnn.utils import ChargeCorrectedNodeWiseAttentiveFP, get_graph_from_mol
from serenityff.charge.gnn.attention_extraction import Extractor
from serenityff.charge.gnn.attention_extraction import Explainer
from serenityff.charge.gnn.attention_extraction.explainer import FixedGNNExplainer
from serenityff.charge.gnn.utils import CustomData
from serenityff.charge.gnn.utils.rdkit_helper import mols_from_sdf
from serenityff.charge.utils import Molecule, command_to_shell_file


@pytest.fixture
def extractor() -> Extractor:
    return Extractor()


@pytest.fixture
def cwd() -> str:
    return os.path.dirname(__file__)


@pytest.fixture
def sdf_path(cwd) -> str:
    return f"{cwd}/../data/example.sdf"


@pytest.fixture
def model_path(cwd) -> str:
    return f"{cwd}/../data/example_model.pt"


@pytest.fixture
def args(sdf_path, statedict_path) -> Sequence[str]:
    return ["-s", sdf_path, "-m", statedict_path]


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
    return Chem.MolToSmiles(mol, canonical=True)


@pytest.fixture
def statedict_path(cwd) -> str:
    return f"{cwd}/../data/example_state_dict.pt"


@pytest.fixture
def statedict(statedict_path) -> OrderedDict:
    return torch.load(statedict_path, map_location=torch.device("cpu"))


@pytest.fixture
def model(statedict) -> ChargeCorrectedNodeWiseAttentiveFP:
    m = ChargeCorrectedNodeWiseAttentiveFP()
    m.load_state_dict(statedict)
    return m


@pytest.fixture
def explainer(model) -> Explainer:
    return Explainer(model=model, epochs=1, verbose=True)


@pytest.fixture
def graph(cwd) -> CustomData:
    return get_graph_from_mol(Chem.SDMolSupplier(f"{cwd}/../data/example.sdf", removeHs=False)[0], index=0)


def test_getter_setter(explainer) -> None:
    with pytest.raises(TypeError):
        explainer.gnn_explainer = "asdf"
    assert isinstance(explainer.gnn_explainer, FixedGNNExplainer)
    return

#Fails due to changes in numpy array_equal
#def test_load(model, statedict) -> None:
#    np.array_equal(model.state_dict(), statedict)
#    return


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


def test_extractor_properties(extractor, model, model_path, statedict_path, explainer) -> None:
    extractor.model = model
    extractor.model = model_path
    extractor.model = statedict_path
    with pytest.raises(TypeError):
        extractor.model = 2
    with pytest.raises(FileNotFoundError):
        extractor.model = "faulty.py"
    with pytest.raises(TypeError):
        extractor.explainer = 2
    extractor.explainer = explainer


def test_split_sdf(cwd, sdf_path) -> None:
    Extractor._split_sdf(
        sdf_file=sdf_path,
        directory=f"{cwd}/sdftest",
    )
    assert os.path.isdir(f"{cwd}/sdftest")
    for file in range(1, 4):
        assert os.path.isfile(f"{cwd}/sdftest/{file}.sdf")
    rmtree(f"{cwd}/sdftest")
    return


def test_job_id(cwd) -> None:
    with open(f"{cwd}/id.txt", "w") as f:
        f.write("sdcep ab ein \n sdf <12345> saoeb <sd>")
    id = Extractor._get_job_id(file=f"{cwd}/id.txt")
    assert id == "12345"
    os.remove(f"{cwd}/id.txt")
    return


def test_mol_from_sdf(sdf_path):
    mol = mols_from_sdf(sdf_file=sdf_path)[0]
    assert mol.GetNumBonds() == 19


def test_graph_from_mol(mol, num_atoms, num_bonds, formal_charge, smiles) -> None:
    graph = get_graph_from_mol(mol=mol, index=0, no_y=True)
    graph = get_graph_from_mol(mol=mol, index=0)
    assert graph.num_nodes == num_atoms
    assert graph.num_edges == num_bonds * 2
    assert graph.edge_attr.shape == torch.Size([38, 11])
    assert graph.molecule_charge.item() == formal_charge
    assert graph.smiles == smiles
    assert graph.x.shape[0] == num_atoms
    assert len(graph.y) == num_atoms
    return


def test_arg_parser(args, sdf_path, statedict_path) -> None:
    args = Extractor._parse_filenames(args)
    assert args.sdffile == sdf_path
    assert args.mlmodel == statedict_path
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


def test_csv_handling(cwd, sdf_path, extractor, model):
    extractor._initialize_expaliner(model=model, epochs=1)
    outfile = f"{cwd}/sdftest/combined.csv"
    Extractor._split_sdf(
        sdf_file=sdf_path,
        directory=f"{cwd}/sdftest",
    )
    for filenr in range(1, 4):
        extractor._explain_molecules_in_sdf(sdf_file=f"{cwd}/sdftest/{filenr}.sdf", scratch=f"{cwd}/sdftest")
    Extractor._summarize_csvs(3, 1, f"{cwd}/sdftest", outfile.rstrip(".csv"))
    df = pd.read_csv(outfile)
    assert df.shape[0] >= 3
    rmtree(f"{cwd}/sdftest")


def test_run_extraction_local(extractor, statedict_path, cwd, sdf_path) -> None:
    extractor.run_extraction_local(
        sdf_file=sdf_path,
        ml_model=statedict_path,
        output=f"{cwd}/combined.csv",
        epochs=1,
    )
    os.path.isfile(f"{cwd}/combined.csv")
    os.remove(f"{cwd}/combined.csv")
    return
