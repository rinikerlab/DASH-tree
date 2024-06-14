"""Functionality to obtain the DASH properties data from ETH research archive."""

from urllib.request import urlretrieve
from pathlib import Path

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

def is_data_complete(folder: Path = ADDITIONAL_DATA_DIR) -> bool:
    """Check if all necessary files are in the according folder.

    Args:
        folder (Path, optional): Folder containing the files and the filelist.txt file. 
        Defaults to ADDITIONAL_DATA_DIR.

    Returns:
        bool: True if all files listed in the 'filelist.txt' are present.
    """
    ...

def get_additional_data(url: str = DATA_URL, zip_archive: Path = ZIP_FILE) -> None:
    """
    Download and extract additional data from ETH research archive.

    Args:
        url (str, optional): _description_. Defaults to DATA_URL.
        zip_archive (Path, optional): _description_. Defaults to ZIP_FILE.
    """
    download_tree_data_from_archive(url=url)
    extract_data(zip_archive=zip_archive)