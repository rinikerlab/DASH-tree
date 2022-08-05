from typing import List
import os
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from serenityff.core.c6_and_partial_charges.tree.atom_features import atom_features


class multi_node:
    def __init__(
        self,
        atom: List = None,
        level: int = None,
        result: float = np.NaN,
        stdDeviation: float = np.NaN,
        attention: float = np.NaN,
        count: int = np.NaN,
        number: int = 0,
    ):
        self.level = level
        self.atoms = atom
        self.hashes = []

        self.result = result
        self.stdDeviation = stdDeviation
        self.attention = attention
        self.count = count
        self.number = number

        self.hashes = self._get_hash()

    def __repr__(self) -> str:
        if self.level == 0:
            return "root"
        return f"mn {len(self.atoms)} - {self.result} - {self.atoms}"

    def __eq__(self, other):
        if isinstance(other, multi_node):
            if other.hashes == self.hashes:
                return True
        else:
            if other.hash in self.hashes:
                for node in self.nodes:
                    if node == other:
                        return True
        return False

    def _get_hash(self):
        return [hash(str(self.level) + str(atom)) for atom in self.atoms]

    def add_node(self, node):
        if isinstance(node, multi_node):
            self.atoms.extend(node.atoms)
            self._update_statistics(node)
            self.hashes.extend(node.hashes)
        else:
            self.atoms.append(node.atom)
            self._update_statistics(node)
            self.hashes.append(node.hash)

        for child in node.children:
            self.add_child(child)

    def _update_statistics(self, other):
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
        if child in self.children:
            idx = self.children.index(child)
            self.children[idx]._update_statistics(child)
            for c in child.children:
                self.add_child(c)
        else:
            self.children.append(child)

    def prune(self, threshold=0.001):
        if hasattr(self, "stdDeviation") and self.stdDeviation != np.nan:
            adjusted_threshold = threshold * ((self.level / 8) + (np.exp(self.level - 20)))

            if self.stdDeviation < adjusted_threshold:
                for child in self.children:
                    if np.abs(child.result - self.result) < adjusted_threshold:
                        self.children.remove(child)

        for child in self.children:
            child.prune(threshold)

    def to_file(self, file):
        node_dict_list = []
        node_dict_list, number = self.node_to_dict(node_dict_list, number=0)
        df = pd.DataFrame(node_dict_list)
        df.to_csv(file)
        print(f"Saved to {file} with {number} nodes")

    def node_to_dict(self, node_dict, number, parent_number=0):
        this_node = {
            "level": self.level,
            "atom": self.atoms,
            "result": self.result,
            "stdDeviation": self.stdDeviation,
            "attention": self.attention,
            "count": self.count,
            "num_children": len(self.children),
            "number": number,
            "parent_number": parent_number,
        }
        node_dict.append(this_node)
        parent_number = number
        number += 1
        for child in self.children:
            node_dict, number = child.node_to_dict(node_dict, number, parent_number)
        return (node_dict, number)

    def from_file(self, file):
        try:
            df = pd.read_csv(
                file,
                index_col=0,
                dtype={
                    "level": np.int32,
                    "atom": str,
                    "result": np.float32,
                    "stdDeviation": np.float32,
                    "attention": np.float32,
                    "count": np.int32,
                    "num_children": np.int32,
                    "number": np.int32,
                },
            )
            self.node_from_df(df, index=0)
        except FileNotFoundError:
            Warning(f"File {file} not found")

    def from_files(self, file_folder, num_workers=1):
        num_files = len(os.listdir(file_folder))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(self.from_file, [file_folder + f"/{i}.csv" for i in range(num_files)])
        self.children = list(results)

    def node_from_df(self, df, index, parent_level=-1):
        if index >= len(df):
            return index
        this_node = df.iloc[index]
        self.level = this_node["level"]
        if self.level <= parent_level:  # saftey to check if node can be child
            return None
        atom_data = this_node["atom"].split(" ") if isinstance(this_node["atom"], str) else []
        self.atom = atom_features(data=atom_data)
        self.result = this_node["result"]
        self.stdDeviation = this_node["stdDeviation"]
        self.attention = this_node["attention"]
        self.count = this_node["count"]
        self.number = this_node["number"]
        self.children = []
        num_children = this_node["num_children"]
        children_found = 0
        if num_children > 0:
            if self.level != 0:
                while children_found < num_children:
                    index += 1
                    child = self.__class__()
                    temp_index = child.node_from_df(df, index, parent_level=self.level)
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
                    temp_index = child.node_from_df(df, index, parent_level=self.level)
                    if temp_index is None:  # handling of safty for children
                        children_found = num_children
                    else:
                        self.children.append(child)
                        children_found += 1
                        index = temp_index
        return index

    def get_attribute_from_df_line(self, df_line):
        self.level = df_line["level"]
        atom_data = df_line["atom"].split(" ") if isinstance(df_line["atom"], str) else []
        self.atom = atom_features(data=atom_data)
        self.result = df_line["result"]
        self.stdDeviation = df_line["stdDeviation"]
        self.attention = df_line["attention"]
        self.count = df_line["count"]
        self.children = []
        try:
            self.num_children = df_line["num_children"]
        except KeyError:
            pass

    def node_is_similar(self, other, min_std_deviation=0.001):
        if self.level in [0, 1, 2, 3]:
            return False
        if abs(self.result - other.result) < min_std_deviation:
            return True
        return False

    def try_to_merge_similar_branches(self, min_std_deviation=0.001, children_overlap_acceptance=0.6):
        for idx, child in enumerate(self.children):
            for other_child in self.children[idx + 1 :]:
                if child.node_is_similar(other_child, min_std_deviation=min_std_deviation):
                    control_bool = True
                    child_match = 0
                    for node_i in child.children:
                        if node_i in other_child.children:
                            child_match += 1
                            if not node_i.node_is_similar(
                                other_child.children[other_child.children.index(node_i)],
                                min_std_deviation=min_std_deviation,
                            ):
                                control_bool = False
                                break
                    if (
                        control_bool
                        and len(child.children) > 0
                        and ((child_match / len(child.children)) >= children_overlap_acceptance)
                    ):
                        child.add_node(other_child)
                        self.children.remove(other_child)
                        return True
            child.try_to_merge_similar_branches(min_std_deviation=min_std_deviation)

    def get_tree_length(self, length_dict):
        length_dict[self.level] += 1
        for child in self.children:
            length_dict = child.get_tree_length(length_dict)
        return length_dict
