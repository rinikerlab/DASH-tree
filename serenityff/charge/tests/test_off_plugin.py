import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField


@pytest.fixture
def force_field():
    return ForceField("openff-2.0.0.offxml")


@pytest.fixture
def force_field_with_plugins():
    return ForceField("openff-2.0.0.offxml", load_plugins=True)


@pytest.fixture
def force_field_custom_offxml():
    return ForceField("serenityff/charge/data/openff-2.0.0-serenity.offxml")


@pytest.fixture
def keys():
    return [
        "Constraints",
        "Bonds",
        "Angles",
        "ProperTorsions",
        "ImproperTorsions",
        "GBSA",
        "vdW",
        "Electrostatics",
        "LibraryCharges",
        "ToolkitAM1BCC",
        "ChargeIncrementModel",
        "VirtualSites",
    ]


@pytest.fixture
def molecule():
    return Molecule.from_smiles("CCO")


def test_off_handler_empty(force_field, keys):
    for key in keys:
        assert force_field.get_parameter_handler(key)
    with pytest.raises(KeyError):
        force_field.get_parameter_handler("SerenityFFCharge")


def test_off_handler_plugins(force_field_with_plugins, keys):
    keys.append("SerenityFFCharge")
    for key in keys:
        assert force_field_with_plugins.get_parameter_handler(key)
    with pytest.raises(KeyError):
        force_field_with_plugins.get_parameter_handler("faulty")


def test_off_handler_custom(force_field_custom_offxml, keys):
    keys.append("SerenityFFCharge")
    for key in keys:
        assert force_field_custom_offxml.get_parameter_handler(key)
    with pytest.raises(KeyError):
        force_field_custom_offxml.get_parameter_handler("faulty")
