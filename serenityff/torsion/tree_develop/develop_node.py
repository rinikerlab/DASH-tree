from typing import Sequence  # , Tuple, Union
import numpy as np

# from serenityff.charge.tree.atom_features import AtomFeatures
from serenityff.charge.tree_develop.develop_node import DevelopNode as DevelopNodeCharge


class DevelopNode(DevelopNodeCharge):
    def __init__(
        self,
        atom_features: list[int, int, int] = None,
        level: int = 0,
        truth_values: Sequence[float] = None,
        attention_values: Sequence[float] = None,
        parent_attention: float = 0,
        num_histogram_bins: int = 100,
    ):
        super().__init__(atom_features, level, truth_values, attention_values, parent_attention)
        self.num_histogram_bins = num_histogram_bins
        self.torsion_histogram = [0] * num_histogram_bins

    def __str__(self) -> str:
        if self.level == 0:
            return f"node --- lvl: {self.level}, Num=1"
        else:
            return f"node --- lvl: {self.level}, Num={str(len(self.truth_values))}, fp={self.atom_features[0]}, {self.atom_features[1]}, {self.atom_features[2]}"

    def add_child(self, child) -> None:
        self.children.append(child)
        return

    def add_node(self, branch: list) -> None:
        current_parent = self
        for node in branch:
            if node in current_parent.children:
                correct_child = current_parent.children[current_parent.children.index(node)]
                correct_child.truth_values.extend(node.truth_values)
                correct_child.attention_values.extend(node.attention_values)
                current_parent = correct_child
            else:
                current_parent.add_child(node)
                current_parent = current_parent.children[-1]
        return

    def _assign_torsion_angle_to_bin(self, angle):
        # angle is between -1 and 1 -1=0, 1=99
        angle = angle + 1
        angle = angle % 2
        bin_size = 2 / self.num_histogram_bins
        bin_index = int(angle / bin_size)
        return bin_index

    def update_average(self):
        if self.level != 0 and self.truth_values is not None:
            for truth_value in self.truth_values:
                bin_index = self._assign_torsion_angle_to_bin(truth_value)
                self.torsion_histogram[bin_index] += 1
        for child in self.children:
            child.update_average()

    def get_node_result_and_std_and_attention_and_length(self, attention_percentage: int = 0.0):
        """
        Returns averaged properties of a node
        Returns the result of the node and the standard deviation of the node and the attention of the node
        Parameters
        ----------
        attention_percentage : int, optional
            threshold for averaging, by default 0.95
        Returns
        -------
        list[float, float, float, int]
            [result, stdDeviation, attention, length]
        """
        self.update_average()
        if self.level == 0:
            return (0, np.nan, np.zeros(self.num_histogram_bins).tolist())
        try:
            return (
                len(self.truth_values),
                np.nanmax(self.attention_values),
                self.torsion_histogram,
            )
        except TypeError:
            return (0, np.nan, np.zeros(self.num_histogram_bins).tolist())

    def get_DASH_data_from_dev_node(self):
        self.update_average()
        if self.level == 0:
            return (0, np.nan, np.nan, np.zeros(self.num_histogram_bins).tolist())
        try:
            return (
                len(self.truth_values),
                np.nanmax(self.attention_values),
                np.nanmean(self.attention_values),
                self.torsion_histogram,
            )
        except TypeError:
            return (0, np.nan, np.nan, np.zeros(self.num_histogram_bins).tolist())
