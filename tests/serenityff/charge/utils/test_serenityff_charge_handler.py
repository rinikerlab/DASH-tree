"""Test serenityff.charge.utils.serenityff_charge_handler.py."""
from pathlib import Path

import pytest
from numpy import allclose
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import (
    ElectrostaticsHandler,
    ForceField,
    LibraryChargeHandler,
    vdWHandler,
)
from packaging.version import Version

from serenityff.charge.tree.dash_tree import DASHTree as Tree
from serenityff.charge.utils.serenityff_charge_handler import SerenityFFChargeHandler

TEST_ATOL = 1e-10
DASH_CHARGES_SOLUTION = [
    -0.49291992,
    0.16308594,
    -0.69091797,
    0.14453125,
    0.14453125,
    0.14453125,
    0.0625,
    0.0625,
    0.4621582,
]


AMBER_CHARGES_SOLUTION = [
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
FF_HANDLER_KEYS = [
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


@pytest.fixture()
def handler() -> SerenityFFChargeHandler:
    return SerenityFFChargeHandler(version=0.3)


@pytest.fixture()
def force_field_with_plugins() -> ForceField:
    return ForceField("openff-2.0.0.offxml", load_plugins=True)


@pytest.fixture()
def force_field_custom_offxml() -> ForceField:
    return ForceField(Path("serenityff/charge/data/openff-2.0.0-serenity.offxml"))


@pytest.fixture
def molecule() -> Molecule:
    return Molecule.from_smiles("CCO")


def test_handler_init(handler: SerenityFFChargeHandler) -> None:
    assert handler._TAGNAME == "SerenityFFCharge"
    assert handler._DEPENDENCIES == [
        ElectrostaticsHandler,
        LibraryChargeHandler,
        vdWHandler,
    ]
    assert handler._KWARGS == ["toolkit_registry"]
    assert handler._MIN_SUPPORTED_SECTION_VERSION == Version("0.3")
    assert handler._MAX_SUPPORTED_SECTION_VERSION == Version("0.3")
    assert handler._SMIRNOFF_VERSIONS == ["0.3"]
    assert handler._ELEMENT_NAME == "SerenityFFCharge"
    assert handler._ELEMENT_CLASSNAME == "SerenityFFChargeHandler"
    assert handler._ELEMENT_DESCRIPTION == "A handler for the SerenityFFCharge parameter tag."
    assert handler._ELEMENTS_PER_TYPE == 1
    assert handler.version == 0.3

    assert isinstance(handler.sff_charge_tree, Tree)
    assert handler.attention_threshold == 10


def test_singleton() -> None:
    instance1 = SerenityFFChargeHandler(version=0.3)
    instance2 = SerenityFFChargeHandler(version=0.3)

    assert instance1 is instance2


def test_loading_off_handler_plugins(
    force_field_custom_offxml: ForceField, force_field_with_plugins: ForceField
) -> None:
    ff = ForceField("openff-2.0.0.offxml", load_plugins=False)
    assert "SerenityFFCharge" not in ff._parameter_handlers
    for key in FF_HANDLER_KEYS:
        assert force_field_with_plugins.get_parameter_handler(key)
        assert force_field_custom_offxml.get_parameter_handler(key)


def test_plugin_charges_get_parameter_handler(
    force_field_with_plugins: SerenityFFChargeHandler,
    molecule,
) -> None:
    assert allclose(
        force_field_with_plugins.get_partial_charges(molecule),
        AMBER_CHARGES_SOLUTION,
        atol=TEST_ATOL,
    )
    force_field_with_plugins.get_parameter_handler("SerenityFFCharge")
    assert allclose(
        force_field_with_plugins.get_partial_charges(molecule),
        DASH_CHARGES_SOLUTION,
        atol=TEST_ATOL,
    )


def test_plugin_charges_register(
    force_field_with_plugins,
    molecule,
    handler,
) -> None:
    assert allclose(
        force_field_with_plugins.get_partial_charges(molecule),
        AMBER_CHARGES_SOLUTION,
        atol=TEST_ATOL,
    )
    force_field_with_plugins.register_parameter_handler(handler)
    assert allclose(
        force_field_with_plugins.get_partial_charges(molecule),
        DASH_CHARGES_SOLUTION,
        atol=TEST_ATOL,
    )


def test_custom_force_field_file_charges(force_field_custom_offxml: ForceField, molecule) -> None:
    assert allclose(
        force_field_custom_offxml.get_partial_charges(molecule),
        DASH_CHARGES_SOLUTION,
        atol=TEST_ATOL,
    )
