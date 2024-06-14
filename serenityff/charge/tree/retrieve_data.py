from urllib.request import urlretrieve
from pathlib import Path

ADDITIONAL_DATA_DIR = Path(__file__).parent.parent/"data"/"additional_data"
ZIP_FILE = ADDITIONAL_DATA_DIR / "dash_data_download.zip"

def download_tree_data_from_archive(url: str = ""):
    return None

def extract_data(path: Path = ZIP_FILE):
    return None