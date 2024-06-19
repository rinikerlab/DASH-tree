import os
from pathlib import Path
from typing import Any, Union

import pytest

from serenityff.charge.data import (
    additional_data_dir,
    default_dash_tree_path,
)
from serenityff.charge.tree.retrieve_data import (
    DataPath,
    data_is_complete,
    download_tree_data_from_archive,
    extract_data,
    get_additional_data,
)
from serenityff.charge.utils.exceptions import DataDownloadError, DataExtractionError
from tests._testfiles import TEST_ARCHIVE


@pytest.fixture()
def tmp_zip(tmp_path: Path) -> Path:
    return tmp_path / "test.zip"


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    tmp_dir = tmp_path / "extracted"
    tmp_dir.mkdir()
    return tmp_dir


def test_paths() -> None:
    """Test that additional data dir exists and is in the right place."""
    assert "/serenityff/charge/data/additional_data" in additional_data_dir.as_posix()
    assert (additional_data_dir / ".gitkeep").exists()


@pytest.mark.parametrize("url", [("invalidurl"), (None), (1234)])
def test_download_tree_data_from_archive_fails(url: Any) -> None:
    """Test that invalid urls actually throw DataDownloadErrors.

    Args:
        url (Any): _description_
    """
    with pytest.raises(DataDownloadError):
        download_tree_data_from_archive(url=url)


@pytest.mark.skipif(
    condition=os.getenv("GITHUB_ACTIONS"),
    reason="Don't download Archive data on github CI.",
)
@pytest.mark.slow
def test_download_tree_data(tmp_zip: Path) -> None:
    """Test that download from default url works.

    Args:
        tmp_zip (Path): where to save file to
    """
    download_tree_data_from_archive(file=tmp_zip)
    assert tmp_zip.exists()
    assert tmp_zip.stat().st_size >= 500000000


@pytest.mark.parametrize(
    "zip_archive", [("notexisting"), (123), (None), Path(__file__)]
)
def test_extract_data_fails(zip_archive: Any) -> None:
    """Test that extraction fails correctly by throwing a DataExtractionError.

    Args:
        zip_archive (Any): wrong files to extract.
    """
    with pytest.raises(DataExtractionError):
        extract_data(zip_archive=zip_archive)


def test_extract_data(tmp_dir: Path) -> None:
    """Test that extracting zip file works as expected.

    Args:
        tmp_dir (Path): tempory directory that exists.
    """
    extract_data(zip_archive=TEST_ARCHIVE, folder=tmp_dir)
    assert data_is_complete(folder=tmp_dir)


@pytest.mark.parametrize(
    "folder, exception", [("faulty", None), (1, TypeError), (None, TypeError)]
)
def test_data_is_complete_fails(folder: Any, exception: Union[None, Exception]) -> None:
    """Test data_is_complete with different input that should fail.

    Args:
        folder (Any): folder to check.
        exception (Union[None, Exception]): should raise this if given otherwise
        `assert not data_is_complete()`.
    """
    if exception is None:
        assert not data_is_complete(folder=folder)
        return
    with pytest.raises(exception):
        data_is_complete(folder=folder)


def test_data_is_complete(tmp_dir: Path) -> None:
    """Test that data_is_complete works for a folder that is complete.

    Args:
        tmp_dir (Path): path where to extract the test zip file to.
    """
    extract_data(zip_archive=TEST_ARCHIVE, folder=tmp_dir)
    assert data_is_complete(folder=tmp_dir)
    assert data_is_complete(folder=default_dash_tree_path)


def test_get_additional_data_already_existing() -> None:
    """Test that get_additional_data returns true when folder already there."""
    assert get_additional_data(extracted_folder=DataPath.DEFAULT)


@pytest.mark.skipif(
    condition=os.getenv("GITHUB_ACTIONS"),
    reason="Don't download Archive data on github CI.",
)
@pytest.mark.slow
def test_get_additional_data_from_archive(tmp_dir: Path) -> None:
    """Test get_additional_data.

    Args:
        tmp_dir (Path): temporary folder to extract data to.
    """
    zipfile = tmp_dir / "test.zip"
    dashpropsdir = tmp_dir / "dashProps"
    assert get_additional_data(
        zip_archive=zipfile, add_data_folder=tmp_dir, extracted_folder=dashpropsdir
    )
    assert dashpropsdir.exists()
    assert data_is_complete(dashpropsdir)
