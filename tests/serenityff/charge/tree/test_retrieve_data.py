from pathlib import Path

import pytest

from serenityff.charge.data import default_dash_tree_path
from serenityff.charge.tree.retrieve_data import (
    ADDITIONAL_DATA_DIR,
    data_is_complete,
    download_tree_data_from_archive,
    extract_data,
    get_additional_data,
)
from serenityff.charge.utils.exceptions import DataDownloadError, DataExtractionError
from tests._testfiles import TEST_ARCHIVE


def test_paths() -> None:
    assert "/serenityff/charge/data" in ADDITIONAL_DATA_DIR.as_posix()
    assert (ADDITIONAL_DATA_DIR / ".gitkeep").exists()


@pytest.mark.parametrize("url", [("invalidurl"), (None)])
def test_download_tree_data_from_archive_fails(url) -> None:
    with pytest.raises(DataDownloadError):
        download_tree_data_from_archive(url=url)


@pytest.mark.parametrize("zip_archive", [("notexisting"), (123), (None)])
def test_extract_data_fails(zip_archive) -> None:
    with pytest.raises(DataExtractionError):
        extract_data(zip_archive=zip_archive)


def test_extract_data(tmp_path: Path) -> None:
    test_folder = tmp_path / "extracted"
    test_folder.mkdir()
    extract_data(zip_archive=TEST_ARCHIVE, folder=test_folder)
    assert data_is_complete(folder=test_folder)


@pytest.mark.parametrize("folder, exception", [("faulty", None), (1, TypeError), (None, TypeError)])
def test_data_is_complete_fails(folder, exception):
    if exception is None:
        assert not data_is_complete(folder)
        return
    with pytest.raises(exception):
        data_is_complete(folder)


def test_data_is_complete(tmp_path: Path):
    test_folder = tmp_path / "extracted"
    test_folder.mkdir()
    extract_data(zip_archive=TEST_ARCHIVE, folder=test_folder)
    assert data_is_complete(folder=test_folder)
    assert data_is_complete(folder=default_dash_tree_path)


def test_get_additional_data_already_existing():
    assert get_additional_data(extracted_folder=default_dash_tree_path)
