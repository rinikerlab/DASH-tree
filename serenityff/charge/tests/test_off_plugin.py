import pytest
from numpy import array_equal
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import (
    ElectrostaticsHandler,
    ForceField,
    LibraryChargeHandler,
    ToolkitAM1BCCHandler,
    vdWHandler,
)

from serenityff.charge.utils.serenityff_charge_handler import SerenityFFChargeHandler


@pytest.fixture
def handler():
    return SerenityFFChargeHandler(version=0.3)


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
        "SerenityFFCharge",
    ]


@pytest.fixture
def molecule():
    return Molecule.from_smiles("CCO")


@pytest.fixture
def charges_serenity():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def charges_amber():
    return [
        -0.13610011111111114,
        0.12639988888888887,
        -0.5998001111111111,
        0.04236688888888887,
        0.04236688888888887,
        0.04236688888888887,
        0.04319988888888887,
        0.04319988888888887,
        0.3959998888888889,
    ]


def test_handler_functions(handler):
    assert handler.check_handler_compatibility(handler) is None
    assert handler._TAGNAME == "SerenityFFCharge"
    assert array_equal(handler._DEPENDENCIES, [ElectrostaticsHandler, LibraryChargeHandler, vdWHandler])
    assert handler._KWARGS == ["toolkit_registry"]


def test_off_handler_empty(force_field, keys):
    for key in keys:
        assert force_field.get_parameter_handler(key)


def test_off_handler_plugins(force_field_with_plugins, keys):
    for key in keys:
        assert force_field_with_plugins.get_parameter_handler(key)
    with pytest.raises(KeyError):
        force_field_with_plugins.get_parameter_handler("faulty")


def test_off_handler_custom(force_field_custom_offxml, keys):
    for key in keys:
        assert force_field_custom_offxml.get_parameter_handler(key)
    with pytest.raises(KeyError):
        force_field_custom_offxml.get_parameter_handler("faulty")


def test_empty_charges(force_field, molecule, handler, charges_amber, charges_serenity):
    assert array_equal(force_field.get_partial_charges(molecule), charges_amber)
    force_field.register_parameter_handler(handler)
    assert array_equal(force_field.get_partial_charges(molecule), charges_serenity)


def test_plugin_charges_get(force_field_with_plugins, molecule, handler, charges_amber, charges_serenity):
    assert array_equal(force_field_with_plugins.get_partial_charges(molecule), charges_amber)
    force_field_with_plugins.get_parameter_handler("SerenityFFCharge")
    assert array_equal(force_field_with_plugins.get_partial_charges(molecule), charges_serenity)


def test_plugin_charges_register(force_field_with_plugins, molecule, handler, charges_amber, charges_serenity):
    assert array_equal(force_field_with_plugins.get_partial_charges(molecule), charges_amber)
    force_field_with_plugins.register_parameter_handler(handler)
    assert array_equal(force_field_with_plugins.get_partial_charges(molecule), charges_serenity)


def test_custom_charges(force_field_custom_offxml, molecule, charges_amber, charges_serenity):
    assert array_equal(force_field_custom_offxml.get_partial_charges(molecule), charges_serenity)
    force_field_custom_offxml.register_parameter_handler(ToolkitAM1BCCHandler(version=0.3))
    assert array_equal(force_field_custom_offxml.get_partial_charges(molecule), charges_serenity)
    force_field_custom_offxml.deregister_parameter_handler("SerenityFFCharge")
    assert array_equal(force_field_custom_offxml.get_partial_charges(molecule), charges_amber)
