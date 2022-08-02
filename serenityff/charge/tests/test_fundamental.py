"""
Unit and regression dev for the serenityff package.

Test most fundamental functionality of package
"""
import sys

import pytest

import serenityff


def test_test() -> None:
    with pytest.raises(AssertionError):
        assert False
    return


def test_sernityff_module_in_sys() -> None:
    assert "serenityff" in sys.modules
    return


def test_serenityff_importable() -> None:
    if serenityff is not None:
        assert True
    else:
        assert False
    return
