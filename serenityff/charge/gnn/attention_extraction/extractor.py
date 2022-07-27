import argparse
import os
import socket
from shutil import make_archive, rmtree
from typing import Optional, OrderedDict, Sequence, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm

from serenityff.charge.gnn.utils import (
    ChargeCorrectedNodeWiseAttentiveFP,
    get_graph_from_mol,
)
from serenityff.charge.utils.io import (
    _get_job_id,
    _split_sdf,
    _summarize_csvs,
    command_to_shell_file,
)

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
                self._model = value
                return
            except AttributeError:
                self._model = ChargeCorrectedNodeWiseAttentiveFP(
                    in_channels=25,
                    hidden_channels=200,
                    out_channels=1,
                    edge_dim=11,
                    num_layers=5,
                    num_timesteps=2,
                )
                self._model.load_state_dict(load)
                return
        elif isinstance(value, torch.nn.Module):
            self._model = value
            return
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
            return
        else:
            raise TypeError(
                "model has to be either of type torch.nn.Module, OrderedDict, \
                    or the str path to a .pt model holding either of the aforementioned types."
            )

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
            graph = get_graph_from_mol(mol=mol)
            graph.to(device="cpu")
            prediction = self.model(
                graph.x,
                graph.edge_index,
                edge_attr=graph.edge_attr,
                batch=graph.batch,
                molecule_charge=graph.molecule_charge,
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
            path_or_buf=f'{scratch}/{sdf_file.split(".")[0].split("/")[-1] + ".csv"}',
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
    def _clean_up(
        num_files: int,
        batch_size: int,
        sdf_file: str,
        local: Optional[bool] = False,
        working_dir: Optional[str] = None,
    ) -> None:
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
        combined_filename = "combined" if not working_dir else f"{working_dir.rstrip('/')}/combined"
        _summarize_csvs(
            num_files=num_files,
            batch_size=batch_size,
            directory=working_dir + "/sdf_data",
            combined_filename=combined_filename,
        )
        if Extractor._check_final_csv(sdf_file=sdf_file, csv_file=combined_filename + ".csv"):
            if not local:
                os.remove("id.txt")
                os.remove("worker.sh")
                os.remove("run_cleanup.sh")
                os.remove("run_extraction.sh")
                os.remove("cleaner.sh")
            make_archive(working_dir + "/sdf_data", "zip", working_dir + "/sdf_data")
            rmtree(working_dir + "/sdf_data/")
        else:
            raise Exception
        return

    @staticmethod
    def run_extraction_local(
        args: Sequence[str],
        working_dir: Optional[str] = None,
        epochs: Optional[int] = 2000,
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
        files = Extractor._parse_filenames(args)
        scratch = "sdf_data" if not working_dir else f"{working_dir.rstrip('/')}/sdf_data"
        num_files, batch_size = _split_sdf(sdf_file=files.sdffile, directory=scratch)
        for file in range(num_files):
            Extractor._extract(
                model=files.mlmodel,
                sdf_index=file + 1,
                working_dir=scratch,
                scratch=scratch,
                verbose=True,
                epochs=epochs,
            )
        Extractor._clean_up(
            num_files=num_files,
            batch_size=batch_size,
            sdf_file=files.sdffile,
            local=True,
            working_dir=working_dir,
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
        num_files, batch_size = _split_sdf(sdf_file=files.sdffile)
        Extractor._write_worker()
        os.mkdir("logfiles")
        lsf_command = f'bsub -n 1 -o logfiles/extraction.out -e logfiles/extraction.err -W 12:00 -J "ext[1-{num_files}]" "./worker.sh {files.mlmodel} {os.getcwd()+"/sdf_data"}" > id.txt'
        command_to_shell_file(lsf_command, "run_extraction.sh")
        os.system(lsf_command)
        id = _get_job_id("id.txt")
        lsf_command = f"bsub -n 1 -J 'clean_up' -o logfiles/cleanup.out -e logfiles/cleanup.err -w 'done({id})' './cleaner.sh {num_files} {batch_size} {files.sdffile}'"
        os.system(lsf_command)
        command_to_shell_file(lsf_command, "run_cleanup.sh")
        return


def main() -> None:
    args = Extractor._parse_filenames()
    host = socket.gethostname()
    if "eu" in host:
        Extractor.run_extraction_lsf(args=args)
    else:
        Extractor.run_extraction_local(args=args, epochs=2000)
    return


if __name__ == "__main__":
    main()
