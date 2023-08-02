import argparse
import os
from math import ceil
from shutil import make_archive, rmtree
from typing import Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

from serenityff.charge.gnn.utils import ChargeCorrectedNodeWiseAttentiveFP, get_graph_from_mol
from serenityff.charge.utils import command_to_shell_file
from serenityff.charge.utils.exceptions import ExtractionError

from .explainer import Explainer


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
            load = torch.load(value, map_location=torch.device("cpu"))
            try:
                load.state_dict()
                self._model = load
                print(self._model.state_dict().keys())
            except AttributeError:
                self._model = ChargeCorrectedNodeWiseAttentiveFP()
                self._model.load_state_dict(load)
        elif isinstance(value, torch.nn.Module):
            self._model = value
        elif isinstance(value, OrderedDict):
            self._model = ChargeCorrectedNodeWiseAttentiveFP(
                in_channels=25,
                hidden_channels=200,
                out_channels=1,
                edge_dim=11,
                num_layers=5,
                num_timesteps=2,
            )
            self._model.load_state_dict(value)
        else:
            raise TypeError(
                "model has to be either of type torch.nn.Module, OrderedDict, \
                    or the str path to a .pt model holding either of the aforementioned types."
            )
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

    def _initialize_expaliner(
        self,
        model: torch.nn.Module,
        epochs: Optional[int] = 1000,
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
        self, sdf_file: str, scratch: str, output: Optional[str] = None, verbose: Optional[bool] = False
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
        for mol_iterator, mol in tqdm(enumerate(suppl), total=len(suppl), disable=not verbose):
            graph = get_graph_from_mol(mol=mol, index=mol_iterator)
            graph.to(device="cpu")
            prediction = self.model(
                graph.x,
                graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=graph.batch,
                molecule_charge=graph.molecule_charge,
            )
            ref_charges = mol.GetProp("MBIScharge").split("|")
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
                        float(ref_charges[atom_iterator]),
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
        out = output if output else f'{scratch}/{sdf_file.split(".")[0].split("/")[-1] + ".csv"}'
        df.to_csv(
            path_or_buf=out,
            index=False,
        )
        return

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
            for atom_iterator, atom in enumerate(mol.GetAtoms()):
                try:
                    if atom_iterator == 0:
                        assert smiles == csv.smiles[csv_iterator]
                    assert mol.GetNumAtoms() == len(csv["node_attentions"][atom_iterator + csv_iterator])
                    assert mol_idx == csv.mol_index[atom_iterator + csv_iterator]
                    assert csv.idx_in_mol[atom_iterator + csv_iterator] == atom_iterator
                    assert csv.atomtype[atom_iterator + csv_iterator] == atom.GetSymbol()
                except AssertionError:
                    is_healthy = False
            csv_iterator += mol.GetNumAtoms()
        return is_healthy

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
    def _split_sdf(
        sdf_file: str, directory: Optional[str] = f"{os.getcwd()}/sdf_data", desiredNumFiles=10000
    ) -> Tuple[int]:
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
        batchsize = ceil(len(suppl) / desiredNumFiles) if len(suppl) > desiredNumFiles else 1
        writer = None
        file_iterator = 0
        for molidx, mol in tqdm(enumerate(suppl), total=len(suppl)):
            if not molidx % batchsize:
                file_iterator += 1
                writer = Extractor._open_next_file(writer=writer, file_name=str(file_iterator), directory=directory)
            writer.write(mol)
        return file_iterator, batchsize

    @staticmethod
    def _get_job_id(file: str, useSlurm=False) -> str:
        """
        Takes a file containing the lsf/slurm submission return and
        extracts the job id of the Job that has been submitted.

        Args:
            file (str): file where the submission return is stored

        Returns:
            str: Jobid
        """
        with open(file, "r") as f:
            txt = f.read()
        if useSlurm:
            id = int(txt.split(" ")[-1].strip())
        else:
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
            try:
                df = pd.read_csv(f"{directory}/{num + 1}.csv")
                df["mol_index"] = df["mol_index"] + num * batch_size
                datalist.append(df)
            except FileNotFoundError:
                print(f"File {num + 1} not found.")
        df = pd.concat(datalist, axis=0, ignore_index=True)
        df.to_csv(combined_filename, index=False)

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
    def _extract_hpc(
        model: Union[str, torch.nn.Module],
        sdf_index: int,
        scratch: str,
        epochs: Optional[int] = 1000,
        working_dir: Optional[str] = None,
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
        sdf_file = f"sdf_data/{sdf_index}.sdf" if not working_dir else f"{working_dir.rstrip('/')}/{sdf_index}.sdf"
        print(scratch, working_dir)
        extractor = Extractor()
        extractor._initialize_expaliner(model=model, epochs=epochs, verbose=verbose)
        extractor._explain_molecules_in_sdf(sdf_file=sdf_file, scratch=scratch)
        return

    @staticmethod
    def _write_worker(directory: Optional[str] = None, useSlurm=False) -> None:
        """
        Writes a bash script called worker.sh, that is then again submitted to the lsf/slurm queue.
        This worker script, does the actual attention extraction on the lsf/slurm hpc cluster.
        """
        file = "worker.sh" if not directory else f"{directory}/worker.sh"
        text = "#!/bin/bash\n"
        text += 'python -c "'
        if useSlurm:
            text += r"from serenityff.charge.gnn.attention_extraction.extractor import Extractor; Extractor._extract_hpc(model='${1}', sdf_index=int(${SLURM_ARRAY_TASK_ID}), scratch='${TMPDIR}')"
            text += '"\n'
            text += r"mv ${TMPDIR}/${SLURM_ARRAY_TASK_ID}.csv ${2}/."
        else:
            text += r"from serenityff.charge.gnn.attention_extraction.extractor import Extractor; Extractor._extract_hpc(model='${1}', sdf_index=int(${LSB_JOBINDEX}), scratch='${TMPDIR}')"
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
        text += r"serenityff.charge.gnn.attention_extraction.extractor import Extractor; Extractor._clean_up(num_files=${1}, batch_size=int(${2}), sdf_file='${3}')"
        text += '"\n'
        with open(file, "w") as f:
            f.write(text)
        os.system(f"chmod u+x {file}")
        return

    @staticmethod
    def _clean_up(
        num_files: int,
        batch_size: int,
        sdf_file: str,
        working_dir: Optional[str] = None,
    ) -> None:
        """
        Cleans up all the messy files after the feature extraction worked flawlessly.

        Args:
            num_files (int): number of smaller .sdf files
            batch_size (int): number of molecules per small .sdf file
            sdf_file (str): original .sdf file

        Raises:
            Exception: Thrown if the generated.csv file is not matching the original .sdf file
        """
        combined_filename = "combined" if not working_dir else f"{working_dir.rstrip('/')}/combined"
        Extractor._summarize_csvs(
            num_files=num_files,
            batch_size=batch_size,
            directory="./sdf_data" if working_dir is None else working_dir + "/sdf_data",
            combined_filename=combined_filename,
        )
        if Extractor._check_final_csv(sdf_file=sdf_file, csv_file=combined_filename + ".csv"):
            for file in [
                "id.txt",
                "worker.sh",
                "run_cleanup.sh",
                "run_extraction.sh",
                "cleaner.sh",
            ]:
                os.remove(file)
            make_archive(working_dir + "/sdf_data", "zip", working_dir + "/sdf_data")
            rmtree(working_dir + "/sdf_data/")
        else:
            raise ExtractionError(
                "Oops Something went wrong with the extraction. \
                Make sure, all paths provided are correct."
            )
        return

    @staticmethod
    def run_extraction_local(
        ml_model=Union[str, torch.nn.Module],
        sdf_file=str,
        output: Optional[str] = "combined.csv",
        epochs: Optional[int] = 1000,
        verbose: Optional[bool] = True,
        verbose_every_atom: Optional[bool] = False,
    ) -> None:
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
        extractor = Extractor()
        extractor._initialize_expaliner(model=ml_model, epochs=epochs, verbose=verbose_every_atom)
        extractor._explain_molecules_in_sdf(sdf_file=sdf_file, output=output, scratch=None, verbose=verbose)
        if not extractor._check_final_csv(sdf_file=sdf_file, csv_file=output):
            raise ExtractionError(
                "Oops Something went wrong with the extraction. \
                Make sure, all paths provided are correct."
            )
        return extractor

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
        files.sdffile = os.path.abspath(files.sdffile.strip())
        print(f"sdf path =|{files.sdffile}|")
        num_files, batch_size = Extractor._split_sdf(sdf_file=files.sdffile)
        Extractor._write_worker()
        if not os.path.exists("logfiles"):
            os.mkdir("logfiles")
        lsf_command = f'bsub -n 1 -o logfiles/extraction.out -e logfiles/extraction.err -W 120:00 -J "ext[1-{num_files}]" "./worker.sh {files.mlmodel} {os.getcwd()+"/sdf_data"}" > id.txt'
        command_to_shell_file(lsf_command, "run_extraction.sh")
        os.system(lsf_command)

        id = Extractor._get_job_id("id.txt")
        Extractor._write_cleaner()
        lsf_command = f"bsub -n 1 -J 'clean_up' -o logfiles/cleanup.out -e logfiles/cleanup.err -w 'done({id})' './cleaner.sh {num_files} {batch_size} {files.sdffile}'"
        os.system(lsf_command)
        command_to_shell_file(lsf_command, "run_cleanup.sh")
        return

    @staticmethod
    def run_extraction_slurm(args: Sequence[str]) -> None:
        """
        Use this function if you want to run the feature extraction on a slurm cluster.

        You need to provide it with an ml model and an .sdf file containing the molecules,
        you want a predcition and attention extractions for.

        Run this static method in a python command and us the following two flags to specify
        all the needed information:
            > -m:   path to the ml model .pt file
            > -s:   path to the .sdf file containing the molecules.

        """
        files = Extractor._parse_filenames(args)
        files.sdffile = os.path.abspath(files.sdffile.strip())
        print(f"sdf path =|{files.sdffile}|")
        num_files, batch_size = Extractor._split_sdf(sdf_file=files.sdffile)
        Extractor._write_worker(useSlurm=True)
        if not os.path.exists("logfiles"):
            os.mkdir("logfiles")
        slurm_command = f'sbatch -n 1 --cpus-per-task=1 --time=120:00:00 --job-name="ext" --array=1-{num_files} --mem-per-cpu=1024 --tmp=64000 --output="logfiles/extraction.out" --error="logfiles/extraction.err" --open-mode=append --wrap="./worker.sh {files.mlmodel} {os.getcwd()+"/sdf_data"}" > id.txt'
        command_to_shell_file(slurm_command, "run_extraction.sh")
        os.system(slurm_command)

        id = Extractor._get_job_id("id.txt", useSlurm=True)
        Extractor._write_cleaner()
        slurm_command = f"sbatch -n 1 --cpus-per-task=1 --time=120:00:00 --job-name='clean_up' --mem-per-cpu=1024 --output='logfiles/cleanup.out' --error='logfiles/cleanup.err' --open-mode=append --dependency=afterok:{id} --wrap='./cleaner.sh {num_files} {batch_size} {files.sdffile}'"
        os.system(slurm_command)
        command_to_shell_file(slurm_command, "run_cleanup.sh")
        return
