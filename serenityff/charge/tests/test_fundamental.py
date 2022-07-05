"""
Unit and regression dev for the serenityff package.

Test most fundamental functionality of package
"""

def test_test():
    assert True

def test_serenityff_importable():
    import serenityff
    assert True

def test_sernityff_module_in_sys():
    import sys
    assert "serenityff" in sys.modules