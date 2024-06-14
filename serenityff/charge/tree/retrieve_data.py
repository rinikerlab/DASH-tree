"""Functionality to obtain the DASH properties data from ETH research archive."""
from urllib.request import urlretrieve
from pathlib import Path

from serenityff.charge.data import default_dash_tree_path
from serenityff.charge.utils.exceptions import DataNotComplete

ADDITIONAL_DATA_DIR = Path(__file__).parent.parent/"data"/"additional_data"
ZIP_FILE = ADDITIONAL_DATA_DIR / "dash_data_download.zip"

DATA_URL = ""

def download_tree_data_from_archive(url: str = DATA_URL) -> None:
    """Download additional DASH Properties data.

    Gets the data uploaded to the ETH-research archive for the DASH-Props
    to fully work.

    Args:
        url (str, optional): Where to get the data. Defaults to "".
    """
    ...

def extract_data(zip_archive: Path = ZIP_FILE) -> None:
    """
    Extract the Downloaded Zip archive to be readable by the DASH-Tree constructor.

    Args:
        zip_archive (Path, optional): Zip archive to extract. Defaults to ZIP_FILE.
    """
    ...

def data_is_complete(folder: Path = ADDITIONAL_DATA_DIR) -> bool:
    """Check if all necessary files are in the according folder.

    Since the atom features dont change, the default tree files as well as the additional
    are exactly the same, so we just need to check whether the newly added ones have the
    same names as the default ones.

    Args:
        folder (Path, optional): Folder containing the files and the filelist.txt file. 
        Defaults to ADDITIONAL_DATA_DIR.

    Returns:
        bool: True if all files listed in the 'filelist.txt' are present.
    """
    for suffix in (".gz", ".h5"):
        default_files = set(file.name for file in default_dash_tree_path.glob(suffix))
        loaded_files = set(file.name for file in ADDITIONAL_DATA_DIR.glob(suffix))
        if len(default_files.intersection(loaded_files)) != len(default_files):
            return False
    return True
    

def get_additional_data(url: str = DATA_URL, zip_archive: Path = ZIP_FILE, folder: Path = ADDITIONAL_DATA_DIR) -> None:
    """Download and extract additional data from ETH research archive.

    Args:
        url (str, optional): where to download from.. Defaults to DATA_URL.
        zip_archive (Path, optional): where to store download. Defaults to ZIP_FILE.
        folder (Path, optional): where to store additional data.

    Raises:
        DataNotComplete: Throw when not all data files necessary where found.
    """
    download_tree_data_from_archive(url=url)
    extract_data(zip_archive=zip_archive)
    if not data_is_complete(folder=folder):
        raise DataNotComplete("Not all files necessary for DASH-Props were found. "
        "Please download and extract them again")