import os
import numpy
import pickle

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Optional
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from matplotlib import lines
from matplotlib import pyplot as plt


def shrink(KeyAppearanceDict: defaultdict, Cutoff: int) -> None:
    """Only keeps keys that appear more often than cutoff

    Args:
        KeyAppearanceDict (Dict): Dict with keys and appearance as the value
        Cutoff (int): Cutoff value.
    """
    for z in list(KeyAppearanceDict.keys()):
        if KeyAppearanceDict[z] <= Cutoff:
            KeyAppearanceDict.pop(z)
    return


def ddict2dict(d: defaultdict) -> dict:
    """Convert defaultdict to dict

    Args:
        d (Defaultdict): Defaultdict to transform

    Returns:
        Dict: Transformed dict
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


class MolMorganDataset:
    """
    Storage of Morgan fingerprints of a molecule dataset (stored in sdf-file). Used to compare to other MolMorganDatasets on a fingerprint count level. And other helpful
    tools for data cleaning, management and merging.
    """

    def __init__(
        self,
        DataPath: str,
        UseOldData: bool = True,
        ConsiderEquivalentKeysOnce: bool = True,
        FilterForChargedMolecules: Optional[Literal["atom", "molecule", "none"]] = "molecule",
    ) -> None:
        """Initializes a new MolMorganDataset and reads in the data from a pickle file, if available, or calculates it new.

        Args:
            DataPath (str): path to sdf file
            UseOldData (bool, optional): If False, data will be calculated new, even if a pickle file is available. Defaults to True.
            ConsiderEquivalentKeysOnce (bool, optional): If False, counts total appearances of fingerprint per molecule. Otherwise each fingerprint is just counted once. Defaults to True.
            FilterForChargedMolecules (Literal, optional): For 'atom' ignores all molecules that contain a charged atom. For 'molecule' ignores all molecules with a formal charge. For 'none' takes all molecules. Default set to 'molecule'
        """
        self._data_path = DataPath
        self._folder_path = os.path.split(self._data_path)[0]
        self._filename = os.path.splitext(os.path.split(self._data_path)[1])[0]  # save file name
        self._saved_path = (
            os.path.split(self._data_path)[0] + "/" + self._filename + "_keys.pkl"
        )  # define path for pickle file
        if not os.path.isfile(self._data_path):  # check if file exists
            raise FileNotFoundError("No dataset was found with the provided name!")
        self._mols = Chem.SDMolSupplier(self._data_path)  # read in molecules
        self._key_dict = defaultdict(
            lambda: defaultdict(
                lambda: 0,
            ),
        )  # stores appearance number for each key. Dicts in a dict for each radius
        self._key_index = defaultdict(
            lambda: 0,
        )  # save molecule of first appearance for each key
        self._key_nocharge_index = defaultdict(
            lambda: 0,
        )
        self._key_dict_nocharge = defaultdict(
            lambda: defaultdict(
                lambda: 0,
            ),
        )
        self._num_mol = None
        self._num_charged_molecule = None
        self._num_not_charged_molecule = None
        self._weights = []
        self._chargedmols = []
        self.ReadData(
            UseOldData=UseOldData,
            FilterForChargedMolecules=FilterForChargedMolecules,
            ConsiderEquivalentKeysOnce=ConsiderEquivalentKeysOnce,
        )  # load data

    def ReadData(
        self,
        UseOldData: bool = True,
        ConsiderEquivalentKeysOnce: bool = True,
        FilterForChargedMolecules: Optional[Literal["atom", "molecule", "none"]] = "molecule",
    ) -> None:
        """Read in data from pickle file if present. Otherwise calls CalcData function to calculate data and store it in a pickle file.

        Args:
            UseOldData (bool, optional): If False, data will be calculated new, even if a pickle file is available. Defaults to True.
            ConsiderEquivalentKeysOnce (bool, optional): If False, counts total appearances of fingerprint per molecule. Otherwise each fingerprint is just counted once. Defaults to True.
            FilterForChargedMolecules (Literal, optional): For 'atom' ignores all molecules that contain a charged atom. For 'molecule' ignores all molecules with a formal charge. For 'none' takes all molecules. Default set to 'molecule'
        """
        if UseOldData and os.path.isfile(self._saved_path):
            print("loaded stored data for " + self._filename)
            # LOAD DATA FROM PICKLE FILE
            with open(self._saved_path, "rb") as handle:  # load pickle file
                inp = pickle.load(handle)
            self._key_dict = inp[0]  # and fill in the variables
            self._key_dict_nocharge = inp[1]
            self._key_index = inp[2]
            self._key_nocharge_index = inp[3]
            self._num_mol = inp[4]
            self._num_charged_molecule = inp[5]
            self._num_not_charged_molecule = self._num_mol - self._num_charged_molecule
            self._weights = inp[6]

        else:  # if no pickle file available calculate data new
            self.CalcData(
                FilterForChargedMolecules=FilterForChargedMolecules,
                ConsiderEquivalentKeysOnce=ConsiderEquivalentKeysOnce,
            )
            print("calculated data for " + self._filename)
        return

    def CalcData(
        self,
        ConsiderEquivalentKeysOnce: bool = True,
        FilterForChargedMolecules: Optional[Literal["atom", "molecule", "none"]] = "molecule",
    ) -> None:
        """Calculate various data for the MolMorganDataset(Appearances of Morgan fingerprints(radius 2), number of charged/uncharged/total molecules, molarweights) and write them into a pickle file

        Args:
            ConsiderEquivalentKeysOnce (bool, optional): If False, counts total appearances of fingerprint per molecule. Otherwise each fingerprint is just counted once. Defaults to True.
            FilterForChargedMolecules (Literal, optional): For 'atom' ignores all molecules that contain a charged atom. For 'molecule' ignores all molecules with a formal charge. For 'none' takes all molecules. Default set to 'molecule'
        """

        prevID = 0
        fp = {}
        atom_charge = 0
        charged = 0
        self._num_not_charged_molecule = 0
        self._num_charged_molecule = 0
        self._num_mol = 0
        for m in self._mols:  # loop over all molecules in MolMorganDataset
            if bool(m.HasProp("CHEMBL_ID")):
                if prevID == m.GetProp("CHEMBL_ID"):
                    self._num_mol += 1  # count amount of molecules. Conformers inlcuded so it later matches the index of molecule list
                    continue
            AllChem.GetMorganFingerprint(m, 2, bitInfo=fp)
            for val in list(fp):  # loop over fingerprints of the molecule
                if not ConsiderEquivalentKeysOnce:
                    self._key_dict[2][val] += len(
                        fp[val]
                    )  # counting appearances of different keys (with taking all equivalent atoms into account)
                    if fp[val][0][1] < 2:  # fingerprints for radius 1
                        self._key_dict[1][val] += len(fp[val])
                    if fp[val][0][1] < 1:  # fingerprints for radius 0
                        self._key_dict[0][val] += len(fp[val])
                else:
                    self._key_dict[2][
                        val
                    ] += 1  # counting appearances of different keys (with counting equivalent atoms as one)
                    if fp[val][0][1] < 2:  # fingerprints for radius 1
                        self._key_dict[1][val] += 1
                    if fp[val][0][1] < 1:  # fingerprints for radius 0
                        self._key_dict[0][val] += 1
                if (
                    self._key_index[val] == 0
                ):  # store molecule where each key appears first --> used if a key should be visualized
                    self._key_index[val] = self._num_mol

            if FilterForChargedMolecules == "atom":  # check if molecule has charged atom
                for atom in m.GetAtoms():
                    atom_charge = atom.GetFormalCharge()
                    if atom_charge != 0:
                        charged = 1
            elif FilterForChargedMolecules == "molecule":  # check if molecule has formal charge
                charged = Chem.GetFormalCharge(m)
            elif FilterForChargedMolecules == "none":
                charged = 0
            else:
                raise ValueError("FilterForChargedMolecules needs to be ['atom', 'molecule', 'none']")
            if charged == 0:  # calculate keys only for uncharged molecules
                for val in list(fp):  # loop over fringerprint key
                    if not ConsiderEquivalentKeysOnce:
                        self._key_dict_nocharge[2][val] += len(
                            fp[val]
                        )  # counting appearances of different keys (with taking all equivalent atoms into account)
                        if fp[val][0][1] < 2:  # fingerprints for radius 1
                            self._key_dict_nocharge[1][val] += len(fp[val])
                        if fp[val][0][1] < 1:  # fingerprints for radius 0
                            self._key_dict_nocharge[0][val] += len(fp[val])
                    else:
                        self._key_dict_nocharge[2][
                            val
                        ] += 1  # counting appearances of different keys (with counting equivalent atoms as one)
                        if fp[val][0][1] < 2:  # fingerprints for radius 1
                            self._key_dict_nocharge[1][val] += 1
                        if fp[val][0][1] < 1:  # fingerprints for radius 0
                            self._key_dict_nocharge[0][val] += 1
                    if (
                        self._key_nocharge_index[val] == 0
                    ):  # store molecule where each key appears first --> used if a key should be visualized
                        self._key_nocharge_index[val] = self._num_mol
                self._num_not_charged_molecule += 1
            else:
                self._num_charged_molecule += 1  # count charged molecules
                self._chargedmols.append(m)
            charged = 0

            if bool(m.HasProp("CHEMBL_ID")):
                prevID = m.GetProp("CHEMBL_ID")  # used to check for conformers

            self._weights.append(Descriptors.ExactMolWt(m))  # calculate weight with H's

            self._num_mol += 1

        # create pickle file with the stored data
        keydict = ddict2dict(self._key_dict)  # transform defaultdicts into dicts so it can be pickled
        keydictnocharge = ddict2dict(self._key_dict_nocharge)
        keyindex = ddict2dict(self._key_index)
        keynochargeindex = ddict2dict(self._key_nocharge_index)
        storable = (
            keydict,
            keydictnocharge,
            keyindex,
            keynochargeindex,
            self._num_mol,
            self._num_charged_molecule,
            self._weights,
        )
        with open(self._saved_path, "wb") as handle:  # write pickle file
            pickle.dump(
                storable, handle, protocol=pickle.HIGHEST_PROTOCOL
            )  # taken from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
        return

    def weight_distribution(self, IgnoreH: bool = False) -> None:
        """Plot the weight distribution

        Args:
            IgnoreH (bool, optional): Wether to include hydrogens to the weight calculation. Defaults to False.
        """

        if not IgnoreH:
            print("heaviest molecule: ", max(self._weights))  # print heaviest and lightest molecule
            print("lightest molecule: ", min(self._weights))
            plt.hist(
                self._weights,
                bins=range(int(numpy.floor(min(self._weights))), int(numpy.ceil(max(self._weights))) + 1, 1),
            )  # plot weightdistribution as scatterplot
            plt.xlabel("Molecular weight [u]")
            plt.ylabel("Counts")
            plt.title("Weight distribution of dataset " + self._filename)
            # plt.axvline(x = 500, color = 'red', linestyle = '--') #to illustrate a cutoff parameter in the plots
            plt.show()

        else:
            weight = []
            for mol in self._mols:  # calculate the weight without H's for each molecule
                mass = 0
                for atom in mol.GetAtoms():  # H's are not stored explicitly in rdkit. So this loop only has non-H atoms
                    if atom.GetAtomicNum != 1:  # just to be sure no H slips through
                        mass += atom.GetMass()
                weight.append(mass)
            print("heaviest molecule: ", max(weight))
            print("lightest molecule: ", min(weight))
            plt.hist(
                weight,
                bins=range(
                    int(numpy.floor(min(weight))) - int(numpy.floor(min(weight))) % 2,
                    int(numpy.ceil(max(weight))) + int(numpy.ceil(max(weight))) % 2,
                    2,
                ),
            )  # make bin width of 2 as it produces substructure for uneven weights otherwise. %2 terms to ensure range is dividable by 2
            plt.xlabel("Molecular weight [u]")
            plt.ylabel("Counts")
            plt.title("Weight distribution of dataset " + self._filename)
            plt.show()
        return

    def intersection(self, otherset) -> set[str]:
        """Calculate the number of molecules that appear in both MolMorganDatasets that are compared

        Args:
            otherset (MolMorganDataset): Used for comparison

        Returns:
            Set of smiles codes that are present in both MolMorganDatasets
        """
        set1 = set(
            [Chem.MolToSmiles(m1, isomericSmiles=False) for m1 in self._mols]
        )  # create set of smiles codes for each molecule
        set2 = set([Chem.MolToSmiles(m2, isomericSmiles=False) for m2 in otherset._mols])
        intersect = set1 & set2  # get intersection between both sets
        print(
            "from the ",
            self._num_mol,
            " molecules in ",
            self._filename,
            " ",
            len(intersect),
            " are also in the set ",
            otherset._filename,
        )
        return intersect  # return list of smiles that are present in both sets

    def compare(self, otherset, UseChargedMolecules: bool = False) -> None:
        """Compares the appearance of morgan fingerprints in MolMorganDatasets and plots them according to the metric (#set1 - #set2)/(#set1 + #set2)

        Args:
            otherset (MolMorganDataset): Used for comparison
            UseChargedMolecules (bool, optional): If False, excludes fingerprints from charged molecules. Defaults to False.
        """
        print("comparing... ")
        self._sort = defaultdict(
            lambda: 0,
        )  # stores difference for keys
        self._color = defaultdict(
            lambda: defaultdict(
                lambda: 0,
            ),
        )  # store color dependent on number of appearances
        self._nosingletons = defaultdict(
            lambda: defaultdict(
                lambda: 0,
            ),
        )  # store differents for keys excluding singletons

        col = {}

        for radius in reversed(range(0, 3)):  # loop over radius 2, 1 and 0
            hold = defaultdict(
                lambda: 0,
            )
            added = defaultdict(
                lambda: 0,
            )
            cnew = defaultdict(
                lambda: 0,
            )
            cold = defaultdict(
                lambda: 0,
            )

            col = {}  # create dictionary to hold the color of the items

            if UseChargedMolecules:
                otherkeys = otherset._key_dict[radius]
                ownkeys = self._key_dict[radius]
            else:
                otherkeys = otherset._key_dict_nocharge[radius]
                ownkeys = self._key_dict_nocharge[radius]

            added = {
                key: otherkeys.get(key, 0)
                + ownkeys.get(key, 0)  # create dictiionary with all keys in old and new set and add their appereances
                for key in set(otherkeys) | set(ownkeys)
            }
            metric = {
                key: (ownkeys.get(key, 0) - otherkeys.get(key, 0))
                / added.get(
                    key, 0
                )  # create dictionary with the calculated metric to measure the difference in the data set: 1 == only appear in new set; -1 == only in old set
                for key in set(otherkeys) | set(ownkeys)
            }
            t = {
                k: v for k, v in sorted(metric.items(), key=lambda item: item[1], reverse=True)
            }  # dict with keys sorted by metric decreasingly
            for i in range(1, 5):  # we want to distinguish between appearances between 1 to >5
                cnew = ownkeys.copy()  # copy dict so we dont permanently delete entries
                cold = otherkeys.copy()
                shrink(cnew, i)  # delete all keys that appear less than i times
                shrink(cold, i)
                add = {
                    key: cold.get(key, 0) + cnew.get(key, 0)  # recalculate difference without deleted items
                    for key in set(cold) | set(cnew)
                }
                add_met = {
                    key: (cnew.get(key, 0) - cold.get(key, 0)) / add.get(key, 0) for key in set(cold) | set(cnew)
                }
                c = {k: v for k, v in sorted(add_met.items(), key=lambda item: item[1], reverse=True)}  # sort by value
                for (k, v) in t.items():  # go through all possible keys
                    if k not in c.keys():  # when they were deleted in this iteration of shrinking ....
                        if k not in hold.keys():  # and they are not yet in included in new...
                            hold[k] = v  # add them
                            col[k] = i - 1  # and store the color depending on the iteration in which they were added
                            # This means elements that only appear once are added first and then the once that appear twice and so on --> they later keep the order
            for (k, v) in t.items():  # finally add all keys that were left over
                if k not in hold.keys():
                    hold[k] = v
                    col[k] = 4
            self._col_string = [
                "firebrick",
                "blue",
                "orange",
                "green",
                "black",
            ]  # red: singletons; blue: keys that only appear twice; orange: 3x; green: 4x; black: more than 5
            self._sort[radius] = {
                k: v for k, v in sorted(hold.items(), key=lambda item: item[1], reverse=True)
            }  # Store sorted dict as entry in another dict where the key is the respective radius
            for k in self._sort[radius].keys():
                self._color[radius][k] = self._col_string[
                    col[k]
                ]  # convert number for color to string with color for later plotting

            # Calculate without any singletons
            for key, value in self._sort[radius].items():
                if added[key] == 1:  # only appear once in one of the data sets
                    continue
                else:
                    if (
                        added[key] == 2 and self._sort[radius][key] == 0
                    ):  # appear twice in total but only in one data set, so they are no singleton
                        continue
                    else:  # appear once in each data set, so they are singletons
                        self._nosingletons[radius][key] = value

    def plot_compare(self, otherset):
        """Executes compare function and plots the data

        Args:
            otherset (MolMorganDataset): Used for comparison
        """
        self.compare(otherset)
        # Create plot of the calculated difference
        fig = plt.figure(constrained_layout=True)
        ax_dict = fig.subplot_mosaic([["A", "B"], ["C", "D"], ["E", "F"]], empty_sentinel="BLANK")

        # Radius 2
        ax_dict["A"].scatter(
            list(range(0, len(self._sort[2]))), list(self._sort[2].values()), color=list(self._color[2].values()), s=0.2
        )  # plot all keys ordered by their difference-value
        ax_dict["A"].hlines(
            y=0, xmin=0, xmax=len(self._sort[2]), color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["A"].vlines(
            x=len(self._sort[2]) / 2.0, ymin=-1, ymax=1, color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["A"].set_title(label="With singletons")
        ax_dict["A"].set_ylabel(ylabel="Fingerpint radius: 2")
        ax_dict["A"].text(x=0, y=0.05, s=self._filename, alpha=0.5, style="italic", fontsize=7.5)
        ax_dict["A"].text(x=0, y=-0.2, s=otherset._filename, alpha=0.5, style="italic", fontsize=7.5)

        ax_dict["B"].scatter(
            list(range(0, len(self._nosingletons[2]))), list(self._nosingletons[2].values()), s=0.2, color="black"
        )  # same plot but without singletons
        ax_dict["B"].hlines(
            y=0, xmin=0, xmax=len(self._nosingletons[2]), color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["B"].vlines(
            x=len(self._nosingletons[2]) / 2.0,
            ymin=-1,
            ymax=1,
            color="black",
            linestyle="dashed",
            alpha=0.7,
            linewidth=0.75,
        )
        ax_dict["B"].set_title(label="Without singletons")
        ax_dict["B"].tick_params(axis="x", labelsize=7.5)

        # Radius 1
        ax_dict["C"].scatter(
            list(range(0, len(self._sort[1]))), list(self._sort[1].values()), color=list(self._color[1].values()), s=0.2
        )
        ax_dict["C"].hlines(
            y=0, xmin=0, xmax=len(self._sort[1]), color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["C"].vlines(
            x=len(self._sort[1]) / 2.0, ymin=-1, ymax=1, color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["C"].set_ylabel(ylabel="Fingerpint radius: 1")

        ax_dict["D"].scatter(
            list(range(0, len(self._nosingletons[1]))), list(self._nosingletons[1].values()), s=0.2, color="black"
        )
        ax_dict["D"].hlines(
            y=0, xmin=0, xmax=len(self._nosingletons[1]), color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["D"].vlines(
            x=len(self._nosingletons[1]) / 2.0,
            ymin=-1,
            ymax=1,
            color="black",
            linestyle="dashed",
            alpha=0.7,
            linewidth=0.75,
        )

        # Radius 0
        ax_dict["E"].scatter(
            list(range(0, len(self._sort[0]))), list(self._sort[0].values()), color=list(self._color[0].values()), s=0.2
        )
        ax_dict["E"].hlines(
            y=0, xmin=0, xmax=len(self._sort[0]), color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["E"].vlines(
            x=len(self._sort[0]) / 2.0, ymin=-1, ymax=1, color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["E"].set_ylabel(ylabel="Fingerpint radius: 0")
        ax_dict["E"].set_xlabel(xlabel="Enumerated keys", fontsize=7.5)

        ax_dict["F"].scatter(
            list(range(0, len(self._nosingletons[0]))), list(self._nosingletons[0].values()), s=0.2, color="black"
        )
        ax_dict["F"].hlines(
            y=0, xmin=0, xmax=len(self._nosingletons[0]), color="black", linestyle="dashed", alpha=0.7, linewidth=0.75
        )
        ax_dict["F"].vlines(
            x=len(self._nosingletons[0]) / 2.0,
            ymin=-1,
            ymax=1,
            color="black",
            linestyle="dashed",
            alpha=0.7,
            linewidth=0.75,
        )
        ax_dict["F"].set_xlabel(xlabel="Enumerated keys", fontsize=7.5)

        # Add legend for line colors
        fig.legend(
            [
                lines.Line2D([0], [0], ls="-", c=self._col_string[0]),
                lines.Line2D([0], [0], ls="-", c=self._col_string[1]),
                lines.Line2D([0], [0], ls="-", c=self._col_string[2]),
                lines.Line2D([0], [0], ls="-", c=self._col_string[3]),
                lines.Line2D([0], [0], ls="-", c=self._col_string[4]),
            ],
            [" = 1", " = 2", " = 3", "= 4", r"$\geq 5$"],
            title="Appearances in data sets",
            title_fontsize=7.5,
            loc="upper center",
            bbox_to_anchor=(0.525, 1.1),
            frameon=True,
            fontsize=5,
        )
        # fig.savefig('singleplot.png', dpi = 1000)
        return

    def missings(
        self, otherset, radius: int = 0, UseChargedMolecules: bool = False, DrawMolecules: bool = False
    ) -> list:
        """Finds the fingerprints are missing that appear in other dataset and prints from how many molecules they origin(this number can be further reduced)

        Args:
            otherset (Dataset): Used for comparison
            radius (int, optional): Radius (maximum of 2) for which the missing fingerprints are calculated. Defaults to 0.
            UseChargedMolecules (bool, optional): If False, excludes fingerprints that come from charged molecules. Defaults to False.
            DrawMolecules (bool, optional): If True, prints a Grid-Image of molecules with missing fingerprints sorted by number of missing fingerprints. Defaults to False.
        Returns:
            List of molecules if DrawMolecules false, otherwise returns a image
        """

        possible_radius = [0, 1, 2]
        if radius not in possible_radius:  # check if radius is acceptable
            raise ValueError("Radius must be 2 or smaller")

        if UseChargedMolecules:
            self._onlys = {
                key: value for (key, value) in otherset._key_dict[radius].items() if key not in self._key_dict[radius]
            }  # keys that only appear in other set
            self._single_onlys = {
                key: value for (key, value) in self._onlys.items() if otherset._key_dict[radius][key] == 1
            }  # only consider singlets of those
            index = otherset._key_index
        else:
            self._onlys = {
                key: value
                for (key, value) in otherset._key_dict_nocharge[radius].items()
                if key not in self._key_dict_nocharge[radius]
            }  # keys that only appear in other set
            self._single_onlys = {
                key: value for (key, value) in self._onlys.items() if otherset._key_dict_nocharge[radius][key] == 1
            }  # only consider singlets of those
            index = otherset._key_nocharge_index
        print("Number of onlys in alternative set: ", len(self._onlys))
        print("Of which ", len(self._single_onlys), " are singletons")

        self._onlymols = defaultdict(
            lambda: [],
        )
        fp = {}
        for key in self._onlys:
            AllChem.GetMorganFingerprint(otherset._mols[index[key]], 2, bitInfo=fp)
            self._onlymols[index[key]].append(
                fp[key][0][0]
            )  # for each index of molecules store from which atom the missing key is produced
        self._onlymols = {
            k: v for (k, v) in sorted(list(self._onlymols.items()), key=lambda t: len(t[1]), reverse=True)
        }  # sort indexes for which one is responsible for the most amount of missing keys
        print("Number of onlys come from ", len(self._onlymols), " different molecules")

        self._missingmols = []
        highlights = []
        fp = {}
        use = 0
        record = defaultdict(
            lambda: 0,
        )
        for key, value in self._onlymols.items():
            mol = otherset._mols[key]
            AllChem.GetMorganFingerprint(mol, 2, bitInfo=fp)
            for fpkey in fp:
                if fpkey in self._onlys and record[fpkey] != 1:  # if key is missing and its not been stored yet...
                    use = 1
                    record[fpkey] = 1
            if use == 1:
                self._missingmols.append(mol)  # store molecule
                highlights.append(value)  # store list of atoms that produce missing key
            use = 0
        print("after filtering the number of needed molecules was reduced to ", len(self._missingmols))

        if DrawMolecules:
            self._missingmols = [
                x for x in self._missingmols if Descriptors.ExactMolWt(x) <= 750
            ]  # only use molecules smaller than 750u. If large ones are included all molecules will be scaled accordingly and are hard to see
            if len(self._missingmols) != 0:
                grid_img = Draw.MolsToGridImage(
                    self._missingmols, molsPerRow=5, subImgSize=(500, 500), highlightAtomLists=highlights
                )  # Draw molecules with missing keys to grid image
                if len(self._missingmols) != len(self._onlymols):
                    print("Some molecules lie under the depiction limit of 750 and were excluded from the image")
                return grid_img
            else:
                print("There are no molecules below the depiction limit of 750 u")
        return self._missingmols

    def add(self, otherset, NewSetName: str) -> None:
        """Add molecules from a new set. Filters out all charged and duplicate molecules.

        Args:
            otherset (Dataset): Dataset that will be added
            NewSetName (str): Name of the new file
        Returns:
            MolMorganDataset: returns the MolMorganDataset that was created
        """
        outputpath = self._folder_path + "/" + NewSetName + ".sdf"
        smiles = defaultdict(
            lambda: [],
        )
        identicals = 0
        with Chem.SDWriter(outputpath) as w:
            for mol in self._mols:  # add all molecules that were in old set
                if Chem.MolToSmiles(mol) not in smiles[len(Chem.MolToSmiles(mol))]:  # make sure there are no duplicates
                    smiles[len(Chem.MolToSmiles(mol))].append(Chem.MolToSmiles(mol))
                    if Chem.GetFormalCharge(mol) == 0:  # make sure they are uncharged
                        w.write(mol)
                else:
                    print("There is a double in old set")
            for mol in otherset._mols:  # add molecules from new set
                if Chem.MolToSmiles(mol) not in smiles[len(Chem.MolToSmiles(mol))]:  # no duplicates
                    smiles[len(Chem.MolToSmiles(mol))].append(Chem.MolToSmiles(mol))
                    if Chem.GetFormalCharge(mol) == 0:  # no charged molecules
                        w.write(mol)
                else:
                    identicals += 1
        print(identicals, " identical molecules were omitted")
        Set = MolMorganDataset(outputpath)
        return Set

    def reduce(self, NewSetName: str, otherset: str = "none", cutoff: int = 5) -> None:
        """Without other set: Deletes molecules with only reduntant fingerprints, that appear more often than the cutoff
        With other set: Adds molecules from other set so that all fingerprints are represented at least as often as the cutoff, if possible.
        (Both use a greedy approach: Molecules that have the highest amount of desired fingerprints are added first.)

        Args:
            NewSetName (str): Name of new file
            otherset (str, optional): Second MolMorganDataset to fill fingerprints. Defaults to 'none'.
            cutoff (int, optional): Value how often each fingerprint should be present. Defaults to 5.
        Returns:
            MolMorganDataset: returns the MolMorganDataset that was created
        """

        fp = {}
        usefull = 0
        smiles = defaultdict(
            lambda: [],
        )
        num = 0
        leftovers = []
        news = 0
        identicals = 0
        keydictnew = defaultdict(
            lambda: 0,
        )
        ranking = {}
        outputpath = self._folder_path + "/" + NewSetName + ".sdf"  # define output path
        print("Newfile is created at path: ", outputpath)
        with Chem.SDWriter(outputpath) as w:
            if (
                otherset == "none"
            ):  # If no other set is given. Reduce number of molecules in set to minimum needed to have all fingerprints at least as often as the cutoff.(If enough are there)
                print("Removing redundants...")
                for (
                    mol
                ) in (
                    self._mols
                ):  # first step: Only keep molecules that contain at least one key appearing 'cutoff' times or less.
                    AllChem.GetMorganFingerprint(mol, 2, bitInfo=fp)
                    for key in fp:
                        if (
                            self._key_dict[2][key] < cutoff
                        ):  # if any key in molecule appears in total less than cutoff, mark as usefull. They are needed no matter what!
                            usefull = 1
                    if usefull == 1:  # write usefull molecules to new file
                        if (
                            Chem.MolToSmiles(mol) not in smiles[len(Chem.MolToSmiles(mol))]
                        ):  # check that its not duplicate
                            if Chem.GetFormalCharge(mol) == 0:  # check that its neutral molecule
                                smiles[len(Chem.MolToSmiles(mol))].append(
                                    Chem.MolToSmiles(mol)
                                )  # store smiles for duplicate detection later
                                w.write(mol)  # write molecule
                                num += 1
                                for key in fp:
                                    keydictnew[key] += 1  # update keydict for next molecules
                        else:
                            identicals += 1
                    else:
                        leftovers.append(mol)  # store leftovers to find smartest ones to add in next step
                    usefull = 0
            else:  # If otherset is given
                print("Completing fingerprints with new set...")
                keydictnew = defaultdict(int, self._key_dict[2])
                leftovers = otherset._mols  # set leftovers as all molecules from other set
                for i, mol in enumerate(self._mols):  # write molecules from old set
                    if Chem.MolToSmiles(mol) not in smiles[len(Chem.MolToSmiles(mol))]:  # check for duplicates
                        if Chem.GetFormalCharge(mol) == 0:  # check that molecule is neutral
                            smiles[len(Chem.MolToSmiles(mol))].append(
                                Chem.MolToSmiles(mol)
                            )  # store smiles for duplicate detection later
                            w.write(mol)  # add molecule
                            num += 1

            for i, mol in enumerate(leftovers):
                AllChem.GetMorganFingerprint(mol, 2, bitInfo=fp)
                for key in fp:
                    if keydictnew[key] < cutoff:
                        news += 1  # count how many keys from each molecule in leftovers has, that are desired for new set(that are still under cutoff)
                    ranking[i] = news
                news = 0
            ranking = {
                k: v for k, v in sorted(ranking.items(), key=lambda item: item[1], reverse=True)
            }  # sort molecules by amount of desired keys --> this is the greedy approach used to solve problem

            for index in ranking:  # go through ranked molecules
                AllChem.GetMorganFingerprint(leftovers[index], 2, bitInfo=fp)
                for key in fp:
                    if keydictnew[key] < cutoff:  # see if it still contributes a new key...
                        usefull = 1
                if usefull == 1:  # if so, ...
                    if (
                        Chem.MolToSmiles(leftovers[index]) not in smiles[len(Chem.MolToSmiles(leftovers[index]))]
                    ):  # check for duplicate...
                        if Chem.GetFormalCharge(leftovers[index]) == 0:  # check for neutrality...
                            smiles[len(Chem.MolToSmiles(leftovers[index]))].append(
                                Chem.MolToSmiles(leftovers[index])
                            )  # store smiles for duplicate detection
                            w.write(leftovers[index])  # add molecuel
                            num += 1
                            for key in fp:
                                keydictnew[key] += 1  # update keydict for molecules in next iterations
                    else:
                        identicals += 1
                usefull = 0
        print(num, " molecules in new set ", NewSetName)
        print(identicals, " identical molecules were omitted")
        Set = MolMorganDataset(outputpath)
        return Set


#    def __repr__():
