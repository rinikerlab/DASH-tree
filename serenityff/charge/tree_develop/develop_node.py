from typing import Sequence, Tuple, Union

import numpy as np

from serenityff.charge.tree.atom_features import AtomFeatures


class DevelopNode:
    def __init__(
        self,
        atom_features: list[int, int, int] = None,
        level: int = 0,
        truth_values: Sequence[float] = None,
        attention_values: Sequence[float] = None,
        parent_attention: float = 0,
    ):
        self.children = []
        self.level = level
        self.parent_attention = parent_attention
        self.average = 0
        self.atom_features = atom_features
        self.truth_values = truth_values
        self.attention_values = attention_values

    def __repr__(self):
        return str(self)

    def __eq__(self, other: object) -> bool:
        return self.level == other.level and (self.atom_features == other.atom_features)

    def __str__(self) -> str:
        if self.level == 0:
            return f"node --- lvl: {self.level}, Num=1"
        else:
            if self.truth_values is None:
                return f"node --- lvl: {self.level}, empty node, fp={AtomFeatures.lookup_int(self.atom_features[0])} ({self.atom_features[1]}, {self.atom_features[2]})"
            return f"node --- lvl: {self.level}, Num={str(len(self.truth_values))}, Mean={float(self.average):.4f}, std={np.std(self.truth_values):.4f}, fp={AtomFeatures.lookup_int(self.atom_features[0])} ({self.atom_features[1]}, {self.atom_features[2]})"

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    def truth_values(self) -> Sequence[float]:
        return self._truth_values

    @property
    def attention_values(self) -> Sequence[float]:
        return self._attention_values

    @property
    def level(self) -> int:
        return self._level

    @property
    def atom_features(self) -> Tuple[int, Tuple[int]]:
        return self._atom_features

    @property
    def parent_attention(self) -> float:
        return self._parent_attention

    @property
    def average(self) -> float:
        return self._average

    @truth_values.setter
    def truth_values(
        self,
        value: Union[Sequence[float], float],
    ) -> None:
        if value is None:
            self._truth_values = None
            return
        if not isinstance(value, list):
            value = [value]
        self._truth_values = value
        return

    @attention_values.setter
    def attention_values(
        self,
        value: Union[Sequence[float], float],
    ) -> None:
        if value is None:
            self._attention_values = None
            return
        if not isinstance(value, list):
            value = [value]
        self._attention_values = value
        return

    @level.setter
    def level(self, value: int) -> None:
        try:
            self._level = int(value)
        except ValueError:
            raise ValueError("Level needs to be of type integer.")
        return

    @atom_features.setter
    def atom_features(self, value: list) -> None:
        self._atom_features = value

    @parent_attention.setter
    def parent_attention(self, value: float) -> None:
        try:
            self._parent_attention = float(value)
        except ValueError:
            raise ValueError("parent attention needs to be of type float")
        return

    @average.setter
    def average(self, value: float) -> None:
        try:
            if isinstance(value, list):
                self._average = np.nanmean(value)
            else:
                self._average = float(value)
        except ValueError:
            raise ValueError("average attention needs to be of type float")
        return

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

    def update_average(self):
        if self.level != 0 and self.truth_values is not None:
            if len(self.truth_values) > 0:
                self.average = np.nanmean(self.truth_values)
            else:
                self.average = np.NAN
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
            return (np.float32(np.nan), np.float32(np.nan), np.float32(np.nan), 0)
        try:
            return (
                np.nanmean(self.truth_values),
                np.nanstd(self.truth_values),
                np.nanmax(self.attention_values),
                len(self.truth_values),
            )
        except TypeError:
            return (np.nan, np.nan, np.nan, 0)
        
    def get_DASH_data_from_dev_node(self):
        self.update_average()
        if self.level == 0:
            return (np.float32(np.nan), np.float32(np.nan), np.float32(np.nan), np.float32(np.nan), 0)
        try:
            return (
                np.nanmean(self.truth_values),
                np.nanstd(self.truth_values),
                np.nanmax(self.attention_values),
                np.nanmean(self.attention_values),
                len(self.truth_values),
            )
        except TypeError:
            return (np.nan, np.nan, np.nan, np.nan, 0)
