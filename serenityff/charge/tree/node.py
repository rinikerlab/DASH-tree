from typing import Dict, List

import numpy as np
import pandas as pd

from serenityff.charge.tree.atom_features import AtomFeatures


class node:
    def __init__(
        self,
        atom: List = [],
        level: int = 0,
        result: float = np.nan,
        stdDeviation: float = np.nan,
        attention: float = np.nan,
        count: int = 0,
        number: int = 0,
        parent_number: int = 0,
    ):
        self.level = level
        self.atoms = None
        if atom is not None:
            if isinstance(atom, int) or isinstance(atom, float):
                self.atoms = [atom]
            else:
                self.atoms = atom
        self.hashes = []

        self.result = result
        self.stdDeviation = stdDeviation
        self.attention = attention
        self.count = count
        self.number = number
        self.parent_number = parent_number
        self.children = []

        if self.atoms is not None:
            self.hashes = self._get_hash()

    def __repr__(self) -> str:
        if self.level == 0:
            return "root"
        return f"node {len(self.atoms)} - {self.result} - {self.count} - {self.atoms}"

    def __eq__(self, other):
        if other.hashes == self.hashes:
            return True
        return False

    def _get_hash(self):
        hashes = []
        if self.atoms is None:
            return hashes
        for atom in self.atoms:
            if atom is None:
                continue
            else:
                hashes.append(atom[0] + 1000 * atom[1] + 1000000 * atom[2])
        return hashes

    def add_node(self, node):
        """
        Add a node to the current node (merge)

        Parameters
        ----------
        node : node
            The node to add to the current node
        """
        self.atoms.extend(node.atoms)
        self._update_statistics(node)
        self.hashes.extend(node.hashes)

        for child in node.children:
            self.add_child(child)

    def _update_statistics(self, other):
        """
        helper function to merge two nodes. Updates the statistics of the current node with the statistics of the other node

        Parameters
        ----------
        other : node
            The node to merge with the current node
        """
        # merge statistics
        # combine result weighted by counts
        self.result = (self.result * self.count + other.result * other.count) / (self.count + other.count)

        # combine stdDeviation weighted by counts
        # had to google that one to figure out how to do this:
        # https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
        if self.count >= 3 and other.count >= 3:
            std_term1 = (
                (self.stdDeviation**2 * (self.count - 1)) + (other.stdDeviation**2 * (other.count - 1))
            ) / (self.count + other.count - 1)
            std_term2 = (self.count * other.count * (self.result - other.result) ** 2) / (
                (self.count + other.count) * (self.count + other.count - 1)
            )
            self.stdDeviation = np.sqrt(std_term1 + std_term2)

        # combine counts
        self.count += other.count

    def add_child(self, child):
        """
        Add a child to the current node

        Parameters
        ----------
        child : node
            The child to add to the current node
        """
        if child in self.children:
            idx = self.children.index(child)
            self.children[idx]._update_statistics(child)
            for c in child.children:
                self.add_child(c)
        else:
            self.children.append(child)

    def fix_nan_stdDeviation(self):
        if hasattr(self, "stdDeviation") and not np.isnan(self.stdDeviation):
            pass
        else:
            self.stdDeviation = 0.05
        for child in self.children:
            child.fix_nan_stdDeviation()

    def prune(self, threshold=0.001):
        """
        Prune the tree to remove nodes with a stdDeviation below a threshold.
        (recursive function)

        Parameters
        ----------
        threshold : float, optional
            pruning threshold, by default 0.001
        """
        # if hasattr(self, "stdDeviation") and self.stdDeviation != np.nan:
        #     adjusted_threshold = threshold * ((self.level / 8) + (np.exp(self.level - 20)))

        #     if self.stdDeviation < adjusted_threshold:
        #         for child in self.children:
        #             if np.abs(child.result - self.result) < adjusted_threshold:
        #                 self.children.remove(child)

        # for child in self.children:
        #     child.prune(threshold)

        if hasattr(self, "stdDeviation") and self.stdDeviation != np.nan:
            adjusted_threshold = threshold * ((self.level / 8))
            # adjusted_threshold = threshold
            if self.stdDeviation < adjusted_threshold:
                for child in self.children:
                    if np.abs(child.result - self.result) < adjusted_threshold:
                        self.children.remove(child)
        else:
            all_children_similar = True
            for child in self.children:
                if np.abs(child.result - self.result) > adjusted_threshold:
                    all_children_similar = False
                    break
            if all_children_similar:
                for child in self.children:
                    self.children.remove(child)
            self.stdDeviation = 0.05

        for child in self.children:
            if child.count < 3 and child.level > 3:
                self.children.remove(child)
            else:
                child.prune(threshold)

    def to_file(self, file_path: str):
        """
        Write the tree to a file.

        Parameters
        ----------
        file : str
            The file to write to
        """
        node_dict_list = []
        node_dict_list, number = self._node_to_dict(node_dict_list, number=0)
        df = pd.DataFrame(node_dict_list)
        df.to_csv(file_path, sep=";", na_rep="NaN")
        print(f"Saved to {file_path} with {number} nodes")

    def _node_to_dict(self, node_dict: Dict, number: int, parent_number: int = 0):
        this_node = {
            "level": np.int32(self.level),
            "atom": str(self.atoms),
            "result": np.float32(self.result),
            "stdDeviation": np.float32(self.stdDeviation),
            "attention": np.float32(self.attention),
            "count": np.int32(self.count),
            "num_children": np.int32(len(self.children)),
            "number": np.int32(number),
            "parent_number": np.int32(parent_number),
        }
        node_dict.append(this_node)
        parent_number = number
        number += 1
        for child in self.children:
            node_dict, number = child._node_to_dict(node_dict, number, parent_number)
        return (node_dict, number)

    def from_file(self, file_path: str, nrows: int = None):
        """
        Read a tree from a file.

        Parameters
        ----------
        file : str
            The file to read from (needs to be csv)
        """
        try:
            df = pd.read_csv(
                file_path,
                index_col=0,
                sep=";",
                dtype={
                    "level": np.int32,
                    "atom": str,
                    "result": np.float32,
                    "stdDeviation": np.float32,
                    "attention": np.float32,
                    "count": np.int32,
                    "num_children": np.int32,
                    "number": np.int32,
                    "parent_number": np.int32,
                },
                nrows=nrows,
            )
            self._node_from_df(df, index=0)
        except FileNotFoundError:
            Warning(f"File {file_path} not found")

    def _df_parse_atoms(self, atoms_line: str):
        self.atoms = []
        try:
            self.atoms = eval(atoms_line)
            assert isinstance(self.atoms, list)
            for atom in self.atoms:
                assert len(atom) == 3
                assert isinstance(atom[0], int)
                assert isinstance(atom[1], int)
                assert isinstance(atom[2], int)
        except Exception as e:
            print(e)
            print(f"Error parsing atoms for node {self.number}\n {atoms_line}")

    def _df_line_parsing(self, df_line: pd.Series):
        self.level = df_line["level"]
        self.result = df_line["result"]
        self.stdDeviation = df_line["stdDeviation"]
        self.attention = df_line["attention"]
        self.count = df_line["count"]
        self.number = df_line["number"]
        self.parent_number = df_line["parent_number"]
        self.children = []
        self._df_parse_atoms(df_line["atom"])

    def _node_from_df(self, df: pd.DataFrame, index: int, parent_level: int = -1):
        if index >= len(df):
            return index
        this_line = df.iloc[index]
        self._df_line_parsing(this_line)
        if self.level <= parent_level:  # saftey to check if node can be child
            return None
        self.children = []
        num_children = this_line["num_children"]
        children_found = 0
        if num_children > 0:
            if self.level != 0:
                while children_found < num_children:
                    index += 1
                    child = self.__class__()
                    temp_index = child._node_from_df(df, index, parent_level=self.level)
                    if temp_index is None:  # handling of safty for children
                        children_found = num_children
                    else:
                        self.children.append(child)
                        children_found += 1
                        index = temp_index
            else:
                while children_found < num_children:
                    index += 1
                    child = self.__class__()
                    temp_index = child._node_from_df(df, index, parent_level=self.level)
                    if temp_index is None:  # handling of safty for children
                        children_found = num_children
                    else:
                        self.children.append(child)
                        children_found += 1
                        index = temp_index
        return index

    def _get_attribute_from_df_line(self, df_line):
        self.level = df_line["level"]
        atom_data = df_line["atom"].split(" ") if isinstance(df_line["atom"], str) else []
        self.atom = AtomFeatures.atom_features_from_data_w_connection_info(data=atom_data)
        self.result = df_line["result"]
        self.stdDeviation = df_line["stdDeviation"]
        self.attention = df_line["attention"]
        self.count = df_line["count"]
        self.children = []
        try:
            self.num_children = df_line["num_children"]
        except KeyError:
            pass

    def node_is_similar(self, other, min_deviation: float = 0.001, af_similar: float = 0.7):
        """
        Check if two nodes are similar (if results are close enough)

        Parameters
        ----------
        other : node
            The other node to compare to
        min_deviation : float, optional
            threhold for similarity, by default 0.001

        Returns
        -------
        bool
            True if similar, False otherwise
        """
        if self.level in [0, 1, 2]:
            return False
        if abs(self.result - other.result) < min_deviation:
            allAFsSimilar = True
            for af1 in self.atoms:
                for af2 in other.atoms:
                    if not AtomFeatures.is_similar_w_connection_info(af1, af2, af_similar):
                        allAFsSimilar = False
                        break
                if not allAFsSimilar:
                    break
            if allAFsSimilar:
                if self.count > other.count:
                    if self.stdDeviation < other.stdDeviation:
                        return True
                else:
                    if self.stdDeviation > other.stdDeviation:
                        return True
        return False

    def try_to_merge_similar_branches(
        self, min_deviation=0.0001, children_overlap_acceptance=0.8, af_similar=0.7, merge_counter=0
    ):
        """
        Try to merge similar branches.

        Parameters
        ----------
        min_deviation : float, optional
            threhold for similarity, by default 0.001
        children_overlap_acceptance : float, optional
            threshold for overlap similarity, by default 0.6
        """
        ret_merge_counter = merge_counter
        for idx, child in enumerate(self.children):
            for other_child in self.children[idx + 1 :]:
                if child.node_is_similar(other_child, min_deviation=min_deviation, af_similar=af_similar):
                    # control_bool = True
                    # child_match = 0
                    # for node_i in child.children:
                    #     if node_i in other_child.children:
                    #         child_match += 1
                    #         if not node_i.node_is_similar(
                    #             other_child.children[other_child.children.index(node_i)],
                    #             min_deviation=min_deviation,
                    #             af_similar=af_similar,
                    #         ):
                    #             control_bool = False
                    #             break
                    # if (
                    #     control_bool
                    #     and len(child.children) > 0
                    #     and ((child_match / len(child.children)) >= children_overlap_acceptance)
                    # ):
                    child.add_node(other_child)
                    self.children.remove(other_child)
                    merge_counter += 1
            ret_merge_counter = child.try_to_merge_similar_branches(
                min_deviation=min_deviation,
                af_similar=af_similar,
                children_overlap_acceptance=children_overlap_acceptance,
                merge_counter=merge_counter,
            )
        return ret_merge_counter

    def get_tree_length(self, length_dict: Dict):
        """
        Get the length of the tree. And store it in a dictionary with level as key.

        Parameters
        ----------
        length_dict : Dict
            Dictionary to store the length of the tree

        Returns
        -------
        Dict
            Dictionary with level as key and length as value
        """
        length_dict[self.level] += 1
        for child in self.children:
            length_dict = child.get_tree_length(length_dict)
        return length_dict
