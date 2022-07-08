"""
Unit and regression dev for the serenityff package.

Test most fundamental functionality of package
"""

import pytest


def test_test():
    with pytest.raises(AssertionError):
        assert False


def test_sernityff_module_in_sys():
    import sys

    assert "serenityff" in sys.modules


def test_serenityff_importable():
    import serenityff

    if serenityff is not None:
        assert True
    else:
        assert False
