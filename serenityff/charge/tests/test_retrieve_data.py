from serenityff.charge.tree.retrieve_data import download_tree_data_from_archive, ADDITIONAL_DATA_DIR
import pytest

def test_paths() -> None:
    assert "/serenityff/charge/data" in ADDITIONAL_DATA_DIR.as_posix()
    assert (ADDITIONAL_DATA_DIR/".gitkeep").exists()

