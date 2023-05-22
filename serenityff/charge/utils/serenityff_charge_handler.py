import numpy as np
from openff.toolkit.topology import TopologyAtom, TopologyVirtualSite
from openff.toolkit.typing.engines.smirnoff import ElectrostaticsHandler, LibraryChargeHandler, vdWHandler
from openff.toolkit.typing.engines.smirnoff.parameters import _NonbondedHandler
from openmm.unit import Quantity, elementary_charge

from serenityff.charge.tree.tree import Tree
from serenityff.charge.data import default_tree_path


class SerenityFFChargeHandler(_NonbondedHandler):

    _TAGNAME = "SerenityFFCharge"
    _DEPENDENCIES = [ElectrostaticsHandler, LibraryChargeHandler, vdWHandler]
    _KWARGS = ["toolkit_registry"]

    sff_charge_tree = Tree()
    attention_threshold = 0.9

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

    def create_force(self, system, topology, **kwargs) -> None:
        force = super().create_force(system, topology, **kwargs)

        # init tree if needed
        if not self.sff_charge_tree.hasData:
            self.sff_charge_tree.from_folder_pickle_lzma(default_tree_path)

        for reference_molecule in topology.reference_molecules:

            for topology_molecule in topology._reference_molecule_to_topology_molecules[reference_molecule]:
                rdkit_mol = reference_molecule.to_rdkit()
                partial_charges = [
                    float(x)
                    for x in self.sff_charge_tree.match_molecule_atoms(
                        mol=rdkit_mol, attention_threshold=self.attention_threshold
                    )[0]
                ]

                for topology_particle in topology_molecule.atoms:
                    if type(topology_particle) is TopologyAtom:
                        ref_mol_particle_index = topology_particle.atom.molecule_particle_index
                    elif type(topology_particle) is TopologyVirtualSite:
                        ref_mol_particle_index = topology_particle.virtual_site.molecule_particle_index
                    else:
                        raise ValueError(f"Particles of type {type(topology_particle)} are not supported")

                    topology_particle_index = topology_particle.topology_particle_index

                    particle_charge = Quantity(partial_charges[ref_mol_particle_index], elementary_charge)

                    _, sigma, epsilon = force.getParticleParameters(topology_particle_index)
                    force.setParticleParameters(topology_particle_index, particle_charge, sigma, epsilon)
                reference_molecule._partial_charges = Quantity(np.array(partial_charges), elementary_charge)
            self.mark_charges_assigned(reference_molecule, topology)
