"""Functionality to obtain the DASH properties data from ETH research archive."""
from urllib.request import urlretrieve
from pathlib import Path
import zipfile

from serenityff.charge.data import default_dash_tree_path
from serenityff.charge.utils.exceptions import DataDownloadError, DataExtractionError

ADDITIONAL_DATA_DIR = Path(__file__).parent.parent / "data" / "additional_data"
ZIP_FILE = ADDITIONAL_DATA_DIR / "dash_data_download.zip"
DATA_URL = ""


def download_tree_data_from_archive(url: str = DATA_URL, file: Path = ZIP_FILE) -> None:
    """Download additional DASH Properties data.

    Gets the data uploaded to the ETH-research archive for the DASH-Props
    to fully work.

    Args:
        url (str, optional): Where to get the data. Defaults to "".
        file (Path, optional): Zip archive to write data to. Defaults to "".
    """
    try:
        print("Downloading data from ETH research archive...")
        urlretrieve(url=url, filename=file)
    except ValueError:
        raise DataDownloadError("The provided Url for the extra data doesn't exist.")
    except TypeError:
        raise DataDownloadError("URL for additional data cannot be None")


def extract_data(zip_archive: Path = ZIP_FILE, folder: Path = ADDITIONAL_DATA_DIR) -> None:
    """
    Extract the Downloaded Zip archive to be readable by the DASH-Tree constructor.

    Args:
        zip_archive (Path, optional): Zip archive to extract. Defaults to ZIP_FILE.
        folder (Path, optional): Where to extract data to. Defaults to ADDITIONAL_DATA_DIR.
    """
    try:
        with zipfile.ZipFile(zip_archive) as zip_ref:
            print("Extracting data...")
            zip_ref.extractall(folder)
    except (FileNotFoundError, AttributeError):
        raise DataExtractionError("Zip to archive was not found.")
    except KeyboardInterrupt:
        pass


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
    if not isinstance(folder, Path):
        folder = Path(folder)
    for suffix in ("*.gz", "*.h5"):
        default_files = set(file.name for file in default_dash_tree_path.glob(suffix))
        loaded_files = set(file.name for file in folder.glob(suffix))
        if len(default_files.intersection(loaded_files)) != len(default_files):
            return False
    return True


def get_additional_data(url: str = DATA_URL, zip_archive: Path = ZIP_FILE, folder: Path = ADDITIONAL_DATA_DIR) -> bool:
    """Download and extract additional data from ETH research archive.

    Args:
        url (str, optional): where to download from.. Defaults to DATA_URL.
        zip_archive (Path, optional): where to store download. Defaults to ZIP_FILE.
        folder (Path, optional): where to store additional data.

    Returns:
        bool: True if files already exist or were downloaded and extracted succesfully.

    Raises:
        DataNotComplete: Throw when not all data files necessary where found.
    """
    if data_is_complete(folder=folder):
        print("Data is already loaded.")
        return True
    download_tree_data_from_archive(url=url)
    extract_data(zip_archive=zip_archive)
    return data_is_complete(folder=folder)
