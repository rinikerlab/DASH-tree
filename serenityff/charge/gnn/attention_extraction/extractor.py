from typing import Optional, List, Union, Tuple, Sequence
from rdkit import Chem
from math import ceil
from shutil import make_archive
from .explainer import Explainer
from serenityff.charge.utils import Molecule
from serenityff.charge.gnn.utils import CustomData, MolGraphConvFeaturizer
from tqdm import tqdm


import argparse
import torch
import os
import pandas as pd
import numpy as np


class Extractor:
    """
    This class handles the whole attention extraction for a trained ml model.
    """

    def __init__(self):
        pass

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, value: Union[str, torch.nn.Module]) -> None:
        if isinstance(value, str):
            self._model = torch.load(value, map_location=torch.device("cpu"))
        else:
            self._model = value
        return

    @property
    def explainer(self) -> Explainer:
        return self._explainer

    @explainer.setter
    def explainer(self, value: Explainer) -> None:
        if not isinstance(value, Explainer):
            raise TypeError("Explainer has to be (derived) of type GNNExplainer from torch_geometric.nn")
        self._explainer = value
        return

    @staticmethod
    def _open_next_file(writer: Chem.SDWriter, file_name: str, directory: str) -> Chem.SDWriter:
        """
        Closes an SDWriter and opens a new one. The new file has the number given
        as file_iterator as its name.

        Args:
            writer (Chem.SDWriter): SDWriter to be closed.
            file_name (int): number added into the new file name.
            directory (str): directory in which the individual sdfs will be stored.

        Returns:
            Chem.SDWriter: New SDWriter
        """
        if writer:
            writer.close()
        if not os.path.isdir(directory):
            os.mkdir(directory)
        writer = Chem.SDWriter(f"{directory}/{file_name}.sdf")
        return writer

    @staticmethod
    def _split_sdf(sdf_file: str, directory: Optional[str] = f"{os.getcwd()}/sdf_data") -> Tuple[int]:
        """
        Splits a big sdf file in a number (<10000) of smaller sdf file,
        to make parallelization on a cluster possible.

        Args:
            sdf_file (str): Big .sdf file to be split
            directory (Optional[str], optional): Where to store the smaller .sdf file. \
                Defaults to f"{os.getcwd()}/sdf_data".

        Returns:
            Tuple[int]: number of files written, number of molecule per file.
        """
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        batchsize = ceil(len(suppl) / 10000)
        writer = None
        file_iterator = 0
        for molidx, mol in tqdm(enumerate(suppl)):
            if not molidx % batchsize:
                file_iterator += 1
                writer = Extractor._open_next_file(writer=writer, file_name=str(file_iterator), directory=directory)
            writer.write(mol)
        return file_iterator, batchsize

    @staticmethod
    def _get_job_id(file: str) -> str:
        """
        Takes a file containing the lsf submission return and
        extracts the job id of the Job that has been submitted.

        Args:
            file (str): file where the submission return is stored

        Returns:
            str: Jobid
        """
        with open(file, "r") as f:
            txt = f.read()
        id = txt.split("<")[1].split(">")[0]
        return id

    @staticmethod
    def _summarize_csvs(
        num_files: int,
        batch_size: int,
        directory: Optional[str] = f"{os.getcwd()}/sdf_data",
        combined_filename: Optional[str] = "combined",
    ) -> None:
        """
        Takes all the csv generated with the extract method and combines
        them to one .csv file containing all attention weights.
        Args:
            num_files (int): number of csv files in the directory.
            batch_size (int): number of molecules per csv file.
            directory (Optional[str], optional): directory where the \
                .sdf files are stored. Defaults to f"{os.getcwd()}/sdf_data".
            combined_filename (Optional[str], optional): name of the final, \
                big .csv file. Defaults to "combined".
        """
        if not combined_filename.endswith(".csv"):
            combined_filename += ".csv"
        datalist = []
        for num in tqdm(range(0, num_files)):
            df = pd.read_csv(f"{directory}/{num + 1}.csv")
            df["mol_idx"] = df["mol_index"] + num * batch_size
            df.drop("mol_index", inplace=True, axis=1)
            datalist.append(df)
        df = pd.concat(datalist, axis=0, ignore_index=True)
        df.to_csv(combined_filename, index=False)

    @staticmethod
    def _check_final_csv(sdf_file: str, csv_file: str) -> bool:
        """
        Performs following checks on the final .csv file:
            > Smile in the first entry of a molecule is same \
                as the one in the original .sfd file.
            > that number of node attentions per atom is \
                equal the number of atoms in the molecule.
            > if the mol_index of a molecule is the same in \
                the final .csv file as in the original .sdf file.
            > if atomnumbering is equal in final .csv and original \
                .sdf file.

        Args:
            sdf_file (str): path to the original .sdf file
            csv_file (str): path to the final .csv file

        Returns:
            bool: Wheter the final.csv checks the tests
        """
        is_healthy = True
        sdf = Chem.SDMolSupplier(sdf_file, removeHs=False)
        csv = pd.read_csv(csv_file)
        csv["node_attentions"] = csv["node_attentions"].apply(eval)
        csv_iterator = 0
        for mol_idx, mol in tqdm(enumerate(sdf), total=len(sdf)):
            smiles = Chem.MolToSmiles(mol, canonical=True)
            for i in range(mol.GetNumAtoms()):
                try:
                    if i == 0:
                        assert smiles == csv.smiles[csv_iterator]
                    assert mol.GetNumAtoms() == len(csv["node_attentions"][i + csv_iterator])
                    assert mol_idx == csv.mol_idx[i + csv_iterator]
                    assert csv.idx_in_mol[i + csv_iterator] == i
                except AssertionError:
                    is_healthy = False
            csv_iterator += mol.GetNumAtoms()
        return is_healthy

    @staticmethod
    def _get_graph_from_mol(
        mol: Molecule,
        allowable_set: Optional[List[str]] = [
            "C",
            "N",
            "O",
            "F",
            "P",
            "S",
            "Cl",
            "Br",
            "I",
            "H",
        ],
    ) -> CustomData:
        """
        Creates an pytorch_geometric Graph from an rdkit molecule.
        The graph contains following features:
            > Node Features:
                > Atom Type (as specified in allowable set)
                > formal_charge
                > hybridization
                > H acceptor_donor
                > aromaticity
                > degree
            > Edge Features:
                > Bond type
                > is in ring
                > is conjugated
                > stereo
        Args:
            mol (Molecule): rdkit molecule
            allowable_set (Optional[List[str]], optional): List of atoms to be \
                included in the feature vector. Defaults to \
                    [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].

        Returns:
            CustomData: pytorch geometric Data with .smiles as an extra attribute.
        """
        grapher = MolGraphConvFeaturizer(use_edges=True)
        graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
        graph.y = torch.tensor(
            [at.GetPropsAsDict()["molFileAlias"] for at in mol.GetAtoms()],
            dtype=torch.float,
        )
        graph.molecule_charge = Chem.GetFormalCharge(mol)
        graph.smiles = Chem.MolToSmiles(mol, canonical=False)
        return graph

    def _initialize_expaliner(
        self,
        model: torch.nn.Module,
        epochs: Optional[int] = 2000,
        verbose: Optional[bool] = False,
    ) -> None:
        """
        Initializes an instance of an Explainer.

        Args:
            model (torch.nn.Module): ml model to explain
            epochs (Optional[int], optional): Epochs used to explain every node. Defaults to 2000.
            verbose (Optional[bool], optional): Wheter to use tqdm progress bars. Defaults to False.
        """
        self.model = model
        self.explainer = Explainer(model=self.model, epochs=epochs, verbose=verbose)

    def _explain_molecules_in_sdf(
        self,
        sdf_file: str,
        scratch: str,
    ) -> None:
        """
        ! Needs an initialized explainer (use initialize_explainer()) !
        Explains the prediction of every atom of every molecule in an .sdf file \
            and saves everythin in an .csv file.

        Args:
            sdf_file (str): sdf containing the molecules to be examined.
            scratch (str): Only used on HPC clusters.
        """
        dataframe = []
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        for mol_iterator, mol in enumerate(suppl):
            graph = Extractor._get_graph_from_mol(mol=mol)
            graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
            graph.to(device="cpu")
            prediction = self.model(
                graph.x,
                graph.edge_index,
                graph.edge_attr,
                graph.batch,
                graph.charge,
            )
            node_attentions, edge_attentions = self.explainer.explain_molecule(graph)
            for atom_iterator, atom in enumerate(mol.GetAtoms()):
                smiles = str(Chem.MolToSmiles(mol)) if atom_iterator == 0 else np.nan
                dataframe.append(
                    [
                        str(atom.GetSymbol()),
                        smiles,
                        int(atom_iterator),
                        int(mol_iterator),
                        node_attentions[atom_iterator].tolist(),
                        edge_attentions[atom_iterator].tolist(),
                        float(prediction.tolist()[atom_iterator][0]),
                        float(float(atom.GetProp("molFileAlias"))),
                    ]
                )

        df = pd.DataFrame(
            dataframe,
            columns=[
                "atomtype",
                "smiles",
                "idx_in_mol",
                "mol_index",
                "node_attentions",
                "edge_attentions",
                "prediction",
                "truth",
            ],
        )
        df.to_csv(
            path_or_buf=f'{scratch}/{sdf_file.split(".")[0].split("/")[1] + ".csv"}',
            index=False,
        )
        return

    @staticmethod
    def _parse_filenames(args: Sequence[str]) -> argparse.Namespace:
        """
        Higly specific, only use as in run_extraction_*.py files.

        Takes the following two flags:
            > -m, --mlmodel:    The MLModel to use. .pt file
            > -s, --sdffile:    .SDF file containing a list of molecules\
                you want a prediction and attention extaction for.

        Returns:
            argparse.Namespace: Namespace containing necessary strings.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--mlmodel", type=str, required=True)
        parser.add_argument("-s", "--sdffile", type=str, required=True)
        return parser.parse_args(args)

    @staticmethod
    def _extract(
        model: Union[str, torch.nn.Module],
        sdf_index: int,
        scratch: str,
        epochs: Optional[int] = 2000,
        verbose: Optional[bool] = False,
    ) -> None:
        """
        Extracts the attention weights a model uses for predictions of \
            the molecules in the sdf.file. Highly specific.

        Args:
            model (Union[str, torch.nn.Module]): Model or path to model to be explained.
            sdf_index (int): index of the .sdf file. Intended to work with split_sdf()
            scratch (str): Only use if working on a HPC Cluster.
            epochs (Optional[int], optional): number of epochs per prediction. Defaults to 2000.
            verbose (Optional[bool], optional): wheter to show tqdm update bar. Defaults to False.
        """
        extractor = Extractor()
        extractor._initialize_expaliner(model=model, epochs=epochs, verbose=verbose)
        extractor._explain_molecules_in_sdf(sdf_file=f"sdf_data/{sdf_index}.sdf", scratch=scratch)
        return

    @staticmethod
    def _write_worker(directory: Optional[str] = None) -> None:
        """
        Writes a bash script called worker.sh, that is then again submitted to the lsf queue.
        This worker script, does the actual attention extraction on the lsf hpc cluster.
        """
        file = "worker.sh" if not directory else f"{directory}/worker.sh"
        text = "#!/bin/bash\n"
        text += 'python -c "'
        text += r"import extractor as e; e.Extractor._extract(model='${1}', sdf_index=int(${LSB_JOBINDEX}), scratch='${TMPDIR}')"
        text += '"\n'
        text += r"mv ${TMPDIR}/${LSB_JOBINDEX}.csv ${2}/."
        with open(file, "w") as f:
            f.write(text)
        os.system(f"chmod u+x {file}")
        return

    @staticmethod
    def _write_cleaner(directory: Optional[str] = None) -> None:
        """
        Writes a basch script called cleaner.sh, that cleans all the mess created by the extractionprocess.
        it runs the Extractor._clean_up() function.
        """
        file = "cleaner.sh" if not directory else f"{directory}/cleaner.sh"
        text = "#!/bin/bash\n"
        text += 'python -c "'
        text += r"import extractor as e; e.Extractor._clean_up(model=${1}, sdf_index=int(${2}), scratch='${3}')"
        text += '"\n'
        with open(file, "w") as f:
            f.write(text)
        os.system(f"chmod u+x {file}")
        return

    @staticmethod
    def _clean_up(num_files: int, batch_size: int, sdf_file: str, local: Optional[bool] = False) -> None:
        """
        Cleans up all the messy files after the feature extraction worked flawlessly.

        Args:
            num_files (int): number of smaller .sdf files
            batch_size (int): number of molecules per small .sdf file
            sdf_file (str): original .sdf file
            local (Optional[bool], optional): Wheter the extraction\
                was run locally or on the lsf cluster. Defaults to False.

        Raises:
            Exception: Thrown if the generated.csv file is not matching the original .sdf file
        """
        Extractor._summarize_csvs(num_files=num_files, batch_size=batch_size)
        if Extractor._check_final_csv(sdf_file=sdf_file, csv_file="combined.csv"):
            if not local:
                os.remove("id.txt")
            make_archive("sdf_data.zip", "zip", "sdf_data")
            os.remove("worker.sh")
            os.remove("run_cleanup.sh")
            os.remove("run_extraction.sh")
        else:
            raise Exception
        return

    @staticmethod
    def _command_to_shell_file(command: str, filename: str) -> None:
        """
        Writes a string to a .sh file

        Args:
            command (str): string to be written
            filename (str): path to file
        """
        with open(filename, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write(command)
        return

    @staticmethod
    def run_extraction_local(args: Sequence[str]) -> None:
        """
        Use this function if you want to run the feature extraction on your local machine.
        Depending on the number of files, this can take up to hours!!!

        You need to provide it with an ml model and an .sdf file containing the molecules,
        you want a predcition and attention extractions for.

        Run this static method in a python command and us the following two flags to specify
        all the needed information:
            > -m:   path to the ml model .pt file
            > -s:   path to the .sdf file containing the molecules.
        """
        files = Extractor._parse_filenames(args)
        num_files, batch_size = Extractor._split_sdf(sdf_file=files.sdffile)
        for file in range(num_files):
            Extractor._extract(
                model=files.mlmodel,
                sdf_index=file + 1,
                scratch="sdf_data",
                verbose=True,
                epochs=10,
            )
        Extractor._clean_up(
            num_files=num_files,
            batch_size=batch_size,
            sdf_file=files.sdffile,
            local=True,
        )
        return

    @staticmethod
    def run_extraction_lsf(args: Sequence[str]) -> None:
        """
        Use this function if you want to run the feature extraction on a lsf cluster.

        You need to provide it with an ml model and an .sdf file containing the molecules,
        you want a predcition and attention extractions for.

        Run this static method in a python command and us the following two flags to specify
        all the needed information:
            > -m:   path to the ml model .pt file
            > -s:   path to the .sdf file containing the molecules.

        """
        files = Extractor._parse_filenames(args)
        num_files, batch_size = Extractor._split_sdf(sdf_file=files.sdffile)
        Extractor._write_worker()
        os.mkdir("logfiles")
        lsf_command = f'bsub -n 1 -o logfiles/extraction.out -e logfiles/extraction.err -W 12:00 -J "ext[1-{num_files}]" "./worker.sh {files.mlmodel} {os.getcwd()+"/sdf_data"}" > id.txt'
        Extractor._command_to_shell_file(lsf_command, "run_extraction.sh")
        os.system(lsf_command)
        id = Extractor._get_job_id("id.txt")
        lsf_command = f"bsub -n 1 -J 'clean_up' -o logfiles/cleanup.out -e logfiles/cleanup.err -w 'done({id})' './cleaner.sh {num_files} {batch_size} {files.sdffile}'"
        os.system(lsf_command)
        Extractor._command_to_shell_file(lsf_command, "run_cleanup.sh")
        return
