"""Functionality to obtain the DASH properties data from ETH research archive."""
import zipfile
from enum import Enum, auto
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

from serenityff.charge.data import (
    additional_data_dir,
    dash_props_tree_path,
    default_dash_tree_path,
)
from serenityff.charge.utils.exceptions import DataDownloadError, DataExtractionError

ZIP_FILE = additional_data_dir / "additional_data_download.zip"


class DataUrl(Enum):
    DEFAULT = auto()
    DASH_PROPS = auto()


class DataPath(Enum):
    DEFAULT = auto()
    DASH_PROPS = auto()


URL_DICT = {
    DataUrl.DEFAULT: None,
    DataUrl.DASH_PROPS: "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/670546/dashProps.zip",
}
DATA_DICT = {
    DataPath.DEFAULT: default_dash_tree_path,
    DataPath.DASH_PROPS: dash_props_tree_path,
}


def download_tree_data_from_archive(url: str = URL_DICT[DataUrl.DASH_PROPS], file: Path = ZIP_FILE) -> None:
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


def extract_data(zip_archive: Path = ZIP_FILE, folder: Path = additional_data_dir) -> None:
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
    except (FileNotFoundError, AttributeError, zipfile.BadZipFile):
        raise DataExtractionError("Zip to archive was not found.")
    except KeyboardInterrupt:
        print("Extraction was interrupted by user.")


def data_is_complete(folder: Path = DATA_DICT[DataPath.DASH_PROPS]) -> bool:
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
    if not folder.exists():
        return False
    for suffix in ("*.gz", "*.h5"):
        default_files = set(file.name for file in default_dash_tree_path.glob(suffix))
        loaded_files = set(file.name for file in folder.glob(suffix))
        if len(default_files.intersection(loaded_files)) != len(default_files):
            return False
    return True


def get_additional_data(
    url: Union[DataUrl, str] = DataUrl.DASH_PROPS,
    zip_archive: Path = ZIP_FILE,
    add_data_folder: Path = additional_data_dir,
    extracted_folder: Union[DataPath] = DataPath.DASH_PROPS,
) -> bool:
    """Download and extract additional data from ETH research archive.

    Args:
        url (str, optional): where to download from.. Defaults to DATA_URL.
        zip_archive (Path, optional): where to store download. Defaults to ZIP_FILE.
        folder (Path, optional): where to store additional data.

    Returns:
        bool: True if files already exist or were downloaded and extracted successfully.

    Raises:
        DataNotComplete: Throw when not all data files necessary where found.
    """
    if not add_data_folder.exists():
        add_data_folder.mkdir()
    if isinstance(url, DataUrl):
        url = URL_DICT[url]
    if isinstance(extracted_folder, DataPath):
        extracted_folder = DATA_DICT[extracted_folder]
    if data_is_complete(folder=extracted_folder):
        return True
    print("The DASH Tree is missing additional data and will install that. This Can take a few minutes...")
    download_tree_data_from_archive(url=url, file=zip_archive)
    extract_data(zip_archive=zip_archive, folder=add_data_folder)
    return data_is_complete(folder=extracted_folder)
