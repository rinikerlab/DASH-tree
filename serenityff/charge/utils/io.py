from rdkit import Chem
import os
from typing import Optional, Tuple
from math import ceil
from tqdm import tqdm
from pandas import read_csv, concat


def command_to_shell_file(command: str, filename: str) -> None:
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
            writer = _open_next_file(writer=writer, file_name=str(file_iterator), directory=directory)
        writer.write(mol)
    return file_iterator, batchsize


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
        df = read_csv(f"{directory}/{num + 1}.csv")
        df["mol_idx"] = df["mol_index"] + num * batch_size
        df.drop("mol_index", inplace=True, axis=1)
        datalist.append(df)
    df = concat(datalist, axis=0, ignore_index=True)
    df.to_csv(combined_filename, index=False)
