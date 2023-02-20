"""_
This file is necessary to eliminate the packages dependecy on deepchem.
Most of the functionality is adapted from deepchem.
https://github.com/deepchem/deepchem/tree/master/deepchem/feat
"""


import inspect
import os
from typing import Any, Iterable, List, Tuple, Union

import numpy as np
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges

from serenityff.charge.utils import Atom, Bond, Molecule

from .custom_data import CustomGraphData

DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]


class Featurizer(object):
    """Abstract class for calculating a set of features for a datapoint.

    This class is abstract and cannot be invoked directly. You'll
    likely only interact with this class if you're a developer. In
    that case, you might want to make a child class which
    implements the `_featurize` method for calculating features for
    a single datapoints if you'd like to make a featurizer for a
    new datatype.
    """

    def featurize(self, datapoints: Iterable[Any], log_every_n: int = 1000, **kwargs) -> np.ndarray:
        """Calculate features for datapoints.

        Parameters
        ----------
        datapoints: Iterable[Any]
        A sequence of objects that you'd like to featurize. Subclassses of
        `Featurizer` should instantiate the `_featurize` method that featurizes
        objects in the sequence.
        log_every_n: int, default 1000
        Logs featurization progress every `log_every_n` steps.

        Returns
        -------
        np.ndarray
        A numpy array containing a featurized representation of `datapoints`.
        """
        datapoints = list(datapoints)
        features = []
        for i, point in enumerate(datapoints):
            if i % log_every_n == 0:
                pass
            try:
                features.append(self._featurize(point, **kwargs))
            except Exception as e:
                print(e)
                features.append(np.array([]))

        return np.asarray(features)

    def __call__(self, datapoints: Iterable[Any], **kwargs):
        """Calculate features for datapoints.

        `**kwargs` will get passed directly to `Featurizer.featurize`

        Parameters
        ----------
        datapoints: Iterable[Any]
        Any blob of data you like. Subclasss should instantiate this.
        """
        return self.featurize(datapoints, **kwargs)

    def _featurize(self, datapoint: Any, **kwargs):
        """Calculate features for a single datapoint.

        Parameters
        ----------
        datapoint: Any
        Any blob of data you like. Subclass should instantiate this.
        """
        raise NotImplementedError("Featurizer is not defined.")

    def __repr__(self) -> str:
        """Convert self to repr representation.

        Returns
        -------
        str
        The string represents the class.

        Examples
        --------
        >>> import deepchem as dc
        >>> dc.feat.CircularFingerprint(size=1024, radius=4)
        CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True,
        features=False, sparse=False, smiles=False]
        >>> dc.feat.CGCNNFeaturizer()
        CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
        """
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != "self"]
        args_info = ""
        for arg_name in args_names:
            value = self.__dict__[arg_name]
            # for str
            if isinstance(value, str):
                value = "'" + value + "'"
            # for list
            if isinstance(value, list):
                threshold = 10
                value = np.array2string(np.array(value), threshold=threshold)
            args_info += arg_name + "=" + str(value) + ", "
        return self.__class__.__name__ + "[" + args_info[:-2] + "]"

    def __str__(self) -> str:
        """Convert self to str representation.

        Returns
        -------
        str
        The string represents the class.

        Examples
        --------
        >>> import deepchem as dc
        >>> str(dc.feat.CircularFingerprint(size=1024, radius=4))
        'CircularFingerprint_radius_4_size_1024'
        >>> str(dc.feat.CGCNNFeaturizer())
        'CGCNNFeaturizer'
        """
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != "self"]
        args_num = len(args_names)
        args_default_values = [None for _ in range(args_num)]
        if args_spec.defaults is not None:
            defaults = list(args_spec.defaults)
            args_default_values[-len(defaults) :] = defaults

        override_args_info = ""
        for arg_name, default in zip(args_names, args_default_values):
            if arg_name in self.__dict__:
                arg_value = self.__dict__[arg_name]
                # validation
                # skip list
                if isinstance(arg_value, list):
                    continue
                if isinstance(arg_value, str):
                    # skip path string
                    if "\\/." in arg_value or "/" in arg_value or "." in arg_value:
                        continue
                # main logic
                if default != arg_value:
                    override_args_info += "_" + arg_name + "_" + str(arg_value)
        return self.__class__.__name__ + override_args_info


class MolecularFeaturizer(Featurizer):
    """Abstract class for calculating a set of features for a
    molecule.

    The defining feature of a `MolecularFeaturizer` is that it
    uses SMILES strings and RDKit molecule objects to represent
    small molecules. All other featurizers which are subclasses of
    this class should plan to process input which comes as smiles
    strings or RDKit molecules.

    Child classes need to implement the _featurize method for
    calculating features for a single molecule.

    Note
    ----
    The subclasses of this class require RDKit to be installed.
    """

    def featurize(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
        """Calculate features for molecules.

        Parameters
        ----------
        datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
        RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
        strings.
        log_every_n: int, default 1000
        Logging messages reported every `log_every_n` samples.

        Returns
        -------
        features: np.ndarray
        A numpy array containing a featurized representation of `datapoints`.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolfiles, rdmolops
            from rdkit.Chem.rdchem import Mol
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if "molecules" in kwargs:
            datapoints = kwargs.get("molecules")
            raise DeprecationWarning('Molecules is being phased out as a parameter, please pass "datapoints" instead.')

        # Special case handling of single molecule
        if isinstance(datapoints, str) or isinstance(datapoints, Mol):
            datapoints = [datapoints]
        else:
            # Convert iterables to list
            datapoints = list(datapoints)

        features: list = []
        for i, mol in enumerate(datapoints):
            if i % log_every_n == 0:
                pass
            try:
                if isinstance(mol, str):
                    # mol must be a RDKit Mol object, so parse a SMILES
                    mol = Chem.MolFromSmiles(mol)
                    # SMILES is unique, so set a canonical order of atoms
                    new_order = rdmolfiles.CanonicalRankAtoms(mol)
                    mol = rdmolops.RenumberAtoms(mol, new_order)

                features.append(self._featurize(mol, **kwargs))
            except Exception:
                if isinstance(mol, Chem.rdchem.Mol):
                    mol = Chem.MolToSmiles(mol)
                features.append(np.array([]))

        return np.asarray(features)


def one_hot_encode(
    val: Union[int, str],
    allowable_set: Union[List[str], List[int]],
    include_unknown_set: bool = False,
) -> List[float]:
    """One hot encoder for elements of a provided set.

    Examples
    --------
    >>> one_hot_encode("a", ["a", "b", "c"])
    [1.0, 0.0, 0.0]
    >>> one_hot_encode(2, [0, 1, 2])
    [0.0, 0.0, 1.0]
    >>> one_hot_encode(3, [0, 1, 2])
    [0.0, 0.0, 0.0]
    >>> one_hot_encode(3, [0, 1, 2], True)
    [0.0, 0.0, 0.0, 1.0]

    Parameters
    ----------
    val: int or str
      The value must be present in `allowable_set`.
    allowable_set: List[int] or List[str]
      List of allowable quantities.
    include_unknown_set: bool, default False
      If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
      An one-hot vector of val.
      If `include_unknown_set` is False, the length is `len(allowable_set)`.
      If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    Raises
    ------
    ValueError
      If include_unknown_set is False and `val` is not in `allowable_set`.
    """

    # init an one-hot vector
    if include_unknown_set is False:
        one_hot_legnth = len(allowable_set)
    else:
        one_hot_legnth = len(allowable_set) + 1
    one_hot = [0.0 for _ in range(one_hot_legnth)]

    try:
        one_hot[allowable_set.index(val)] = 1.0  # type: ignore
    except ValueError:
        if include_unknown_set:
            # If include_unknown_set is True, set the last index is 1.
            one_hot[-1] = 1.0
        else:
            pass
    return one_hot


def get_atom_hydrogen_bonding_one_hot(atom: Atom, hydrogen_bonding: List[Tuple[int, str]]) -> List[float]:
    """Get an one-hot feat about whether an atom accepts electrons or donates electrons.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object
    hydrogen_bonding: List[Tuple[int, str]]
      The return value of `construct_hydrogen_bonding_info`.
      The value is a list of tuple `(atom_index, hydrogen_bonding)` like (1, "Acceptor").

    Returns
    -------
    List[float]
      A one-hot vector of the ring size type. The first element
      indicates "Donor", and the second element indicates "Acceptor".
    """
    one_hot = [0.0, 0.0]
    atom_idx = atom.GetIdx()
    for hydrogen_bonding_tuple in hydrogen_bonding:
        if hydrogen_bonding_tuple[0] == atom_idx:
            if hydrogen_bonding_tuple[1] == "Donor":
                one_hot[0] = 1.0
            elif hydrogen_bonding_tuple[1] == "Acceptor":
                one_hot[1] = 1.0
    return one_hot


def get_atom_total_degree_one_hot(
    atom: Atom,
    allowable_set: List[int] = [0, 1, 2, 3, 4, 5],
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of the degree which an atom has.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object
    allowable_set: List[int]
      The degree to consider. The default set is `[0, 1, ..., 5]`
    include_unknown_set: bool, default True
      If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
      A one-hot vector of the degree which an atom has.
      If `include_unknown_set` is False, the length is `len(allowable_set)`.
      If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(atom.GetTotalDegree(), allowable_set, include_unknown_set)


def get_atom_partial_charge(atom: Atom) -> List[float]:
    """Get a partial charge of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object

    Returns
    -------
    List[float]
      A vector of the parital charge.

    Notes
    -----
    Before using this function, you must calculate `GasteigerCharge`
    like `AllChem.ComputeGasteigerCharges(mol)`.
    """
    gasteiger_charge = atom.GetProp("_GasteigerCharge")
    if gasteiger_charge in ["-nan", "nan", "-inf", "inf"]:
        gasteiger_charge = 0.0
    return [float(gasteiger_charge)]


def _construct_atom_feature(
    atom: Atom,
    h_bond_infos: List[Tuple[int, str]],
    use_partial_charge: bool,
    allowable_set: List[str],
) -> np.ndarray:
    """
    Constructs an atom feature from a RDKit atom object.
    In this case creates one hot features for the Attentive FP model.
    The only thing changed is, that it now passes information about
    what atom types you want to have a feature for, given in allowable_set.

    Args:
        atom (RDKitAtom): RDKit atom object
        h_bond_infos (List[Tuple[int, str]]): A list of tuple `(atom_index, hydrogen_bonding_type)`.
        Basically, it is expected that this value is the return value
        of `construct_hydrogen_bonding_info`.The `hydrogen_bonding_type` value is "Acceptor"
        or "Donor".use_partial_charge (bool): Whether to use partial charge data or not.
        allowable_set (List[str]): List of Atoms you want to have features for.

    Returns:
        np.ndarray: A one-hot vector of the atom feature.
    """
    atom_type = one_hot_encode(atom.GetSymbol(), allowable_set, True)
    formal_charge = [float(atom.GetFormalCharge())]
    hybridization = one_hot_encode(str(atom.GetHybridization()), DEFAULT_HYBRIDIZATION_SET, False)
    # remove
    # acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = [float(atom.GetIsAromatic())]
    degree = one_hot_encode(atom.GetTotalDegree(), DEFAULT_TOTAL_DEGREE_SET, True)
    atom_feat = np.concatenate(
        [
            atom_type,
            formal_charge,
            hybridization,
            #    acceptor_donor,
            aromatic,
            degree,
        ]
    )
    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])
        atom_feat = np.concatenate([atom_feat, np.array(atom.Get)])
    return atom_feat


def _construct_bond_feature(bond: Bond) -> np.ndarray:
    """
    Construct a bond feature from a RDKit bond object. Not changed.

    Args:
        bond (RDKitBond): RDKit bond object

    Returns:
        np.ndarray: A one-hot vector of the bond feature.
    """
    bond_type = one_hot_encode(str(bond.GetBondType()), DEFAULT_BOND_TYPE_SET, False)
    same_ring = [int(bond.IsInRing())]
    conjugated = [int(bond.GetIsConjugated())]
    stereo = one_hot_encode(str(bond.GetStereo()), DEFAULT_BOND_STEREO_SET, True)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


class _ChemicalFeaturesFactory:
    """This is a singleton class for RDKit base features."""

    _instance = None

    @classmethod
    def get_instance(cls):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if not cls._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
        return cls._instance


def construct_hydrogen_bonding_info(mol: Molecule) -> List[Tuple[int, str]]:
    """Construct hydrogen bonding infos about a molecule.

    Parameters
    ---------
    mol: rdkit.Chem.rdchem.Mol
      RDKit mol object

    Returns
    -------
    List[Tuple[int, str]]
      A list of tuple `(atom_index, hydrogen_bonding_type)`.
      The `hydrogen_bonding_type` value is "Acceptor" or "Donor".
    """
    factory = _ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding = []
    for f in feats:
        hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))
    return hydrogen_bonding


class MolGraphConvFeaturizer(MolecularFeaturizer):
    """
    Same as original by deepchem, excyept, that you now can give an allowable set,
    that determines for which atom types a feature in the one hot vector is created.


    This class is a featurizer of general graph convolution networks for molecules.
    The default node(atom) and edge(bond) representations are based on
    `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_.
    If you want to use your own representations, you could use this class as a guide
    to define your original Featurizer. In many cases, it's enough
    to modify return values of `construct_atom_feature` or `construct_bond_feature`.
    The default node representation are constructed by concatenating the following values,
    and the feature length is 30.
    - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other".
    - Formal charge: Integer electronic charge.
    - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
    - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
    - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
    - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
    - Partial charge: Calculated partial charge. (Optional)
    The default edge representation are constructed by concatenating the following values,
    and the feature length is 11.
    - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
    - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
    - Conjugated: A one-hot vector of whether this bond is conjugated or not.
    - Stereo: A one-hot vector of the stereo configuration of a bond.
    If you want to know more details about features, please check the paper [1]_ and
    utilities in deepchem.utils.molecule_feature_utils.py.
    Examples
    --------
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.CustomGraphData'>
    >>> out[0].num_node_features
    30
    >>> out[0].num_edge_features
    11
    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
       Journal of computer-aided molecular design 30.8 (2016):595-608.
    Note
    ----
    This class requires RDKit to be installed.
    """

    def __init__(
        self,
        use_edges: bool = False,
        use_partial_charge: bool = False,
    ):
        """
        Parameters
        ----------
        use_edges: bool, default False
          Whether to use edge features or not.
        use_partial_charge: bool, default False
          Whether to use partial charge data or not.
          If True, this featurizer computes gasteiger charges.
          Therefore, there is a possibility to fail to featurize for some molecules
          and featurization becomes slow.
        """
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge

    def _featurize(self, datapoint: Molecule, allowable_set: List[str], **kwargs) -> CustomGraphData:
        """Calculate molecule graph features from RDKit mol object.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        allowable_set: List[str]
          List of atoms you want a feature for in the atom feature vector
        Returns
        -------
        graph: CustomGraphData
          A molecule graph with some features.
        """
        assert (
            datapoint.GetNumAtoms() > 1
        ), "More than one atom should be present in the molecule for this featurizer to work."
        if "mol" in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning('Mol is being phased out as a parameter, please pass "datapoint" instead.')

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp("_GasteigerCharge")
            except KeyError:
                # If partial charges were not computed
                try:
                    ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        # h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        h_bond_infos = [(i, "Donor") for i in range(datapoint.GetNumAtoms())]
        atom_features = np.asarray(
            [
                _construct_atom_feature(
                    atom,
                    h_bond_infos,
                    self.use_partial_charge,
                    allowable_set=allowable_set,
                )
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        return CustomGraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features,
        )
