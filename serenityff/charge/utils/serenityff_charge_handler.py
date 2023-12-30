from typing import List
import numpy as np

# from openff.toolkit.topology import TopologyAtom, TopologyVirtualSite
from openff.toolkit.typing.engines.smirnoff import (
    ElectrostaticsHandler,
    LibraryChargeHandler,
    vdWHandler,
    #    ToolkitAM1BCCHandler,
)
from openff.toolkit.typing.engines.smirnoff.parameters import _NonbondedHandler
from openff.toolkit.utils.base_wrapper import ToolkitWrapper
from openmm.unit import Quantity, elementary_charge
from packaging.version import Version

# from serenityff.charge.tree.tree import Tree
# from serenityff.charge.data import default_tree_path
from serenityff.charge.tree.dash_tree import DASHTree as Tree


class SerenityFFChargeHandler(_NonbondedHandler, ToolkitWrapper):

    _TAGNAME = "SerenityFFCharge"
    _DEPENDENCIES = [ElectrostaticsHandler, LibraryChargeHandler, vdWHandler]
    _KWARGS = ["toolkit_registry"]
    _MIN_SUPPORTED_SECTION_VERSION = Version("0.3")
    _MAX_SUPPORTED_SECTION_VERSION = Version("0.3")
    _SMIRNOFF_VERSIONS = ["0.3"]
    _ELEMENT_NAME = "SerenityFFCharge"
    _ELEMENT_CLASSNAME = "SerenityFFChargeHandler"
    _ELEMENT_DESCRIPTION = "A handler for the SerenityFFCharge parameter tag."
    _ELEMENTS_PER_TYPE = 1
    version = "0.3"

    sff_charge_tree = Tree()
    attention_threshold = 10

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            cls.instance = super(SerenityFFChargeHandler, cls).__new__(cls)
        return cls.instance

    def check_handler_compatibility(self, other_handler, assume_missing_is_default=True):
        """
        Checks whether this ParameterHandler encodes compatible physics as another ParameterHandler. This is
        called if a second handler is attempted to be initialized for the same tag.
        Parameters
        ----------
        other_handler : a ParameterHandler object
            The handler to compare to.
        Raises
        ------
        IncompatibleParameterError if handler_kwargs are incompatible with existing parameters.
        """
        pass

    def assign_partial_charges(self, molecule, **kwargs) -> List[Quantity]:
        print(type(molecule))
        if hasattr(molecule, "to_rdkit"):
            rdkit_mol = molecule.to_rdkit()
        else:
            rdkit_mol = molecule.reference_molecule.to_rdkit()
        partial_charges = [
            float(x)
            for x in self.sff_charge_tree.get_molecules_partial_charges(
                mol=rdkit_mol, attention_threshold=self.attention_threshold
            )["charges"]
        ]
        partial_charges_with_units = Quantity(np.array(partial_charges), unit=elementary_charge)
        # charges = unit.Quantity(charges, unit.elementary_charge)
        molecule.partial_charges = partial_charges_with_units
        print(f"Assigned partial charges: {partial_charges_with_units}")
        return partial_charges_with_units

    def create_force(self, system, topology, **kwargs) -> None:
        force = super().create_force(system, topology, **kwargs)

        for reference_molecule in topology.reference_molecules:

            for topology_molecule in topology._reference_molecule_to_topology_molecules[reference_molecule]:
                partial_charges = self.assign_partial_charges(topology_molecule)

                for topology_particle in topology_molecule.atoms:
                    try:
                        ref_mol_particle_index = topology_particle.atom.molecule_particle_index
                    except AttributeError:
                        try:
                            ref_mol_particle_index = topology_particle.virtual_site.molecule_particle_index
                        except AttributeError:
                            raise ValueError(f"Particles of type {type(topology_particle)} are not supported")
                    # if type(topology_particle) is TopologyAtom:
                    #     ref_mol_particle_index = topology_particle.atom.molecule_particle_index
                    # elif type(topology_particle) is TopologyVirtualSite:
                    #     ref_mol_particle_index = topology_particle.virtual_site.molecule_particle_index
                    # else:
                    #     raise ValueError(f"Particles of type {type(topology_particle)} are not supported")

                    topology_particle_index = topology_particle.topology_particle_index

                    particle_charge = partial_charges[ref_mol_particle_index]

                    _, sigma, epsilon = force.getParticleParameters(topology_particle_index)
                    force.setParticleParameters(topology_particle_index, particle_charge, sigma, epsilon)
                reference_molecule._partial_charges = partial_charges
            self.mark_charges_assigned(reference_molecule, topology)
