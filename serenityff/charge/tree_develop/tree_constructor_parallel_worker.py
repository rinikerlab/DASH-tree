# import datetime
import numpy as np
from serenityff.charge.tree.atom_features import AtomFeatures

from serenityff.charge.tree.tree_utils import (
    get_connected_atom_with_max_attention,
    get_connected_neighbor,
    get_possible_connected_new_atom,
)
from serenityff.charge.tree_develop.develop_node import DevelopNode

try:
    from multiprocessing import Pool
except ImportError:
    print("multiprocessing not installed, parallelization not possible")


class Tree_constructor_parallel_worker:
    def __init__(
        self,
        df_af_split,
        matrices,
        feature_dict,
        roots,
        bond_matrices,
        num_layers_to_build,
        attention_percentage,
        verbose=False,
        loggingBuild=False,
    ):
        self.df_af_split = df_af_split
        self.matrices = matrices
        self.feature_dict = feature_dict
        self.roots = roots
        self.bond_matrices = bond_matrices
        self.num_layers_to_build = num_layers_to_build
        self.attention_percentage = attention_percentage
        self.verbose = verbose
        self.loggingBuild = loggingBuild
        # self.dask_client = Client()
        # if verbose:
        #    print(self.dask_client)

    def _match_atom(self, layer, line):
        current_node = self.roots[line["atom_feature"]]
        for i in range(layer):
            for child in current_node.children:
                if child.atom_features == line[i]:
                    current_node = child
                    break
        return current_node

    def _find_matching_child(self, children, matrix, indices, mol_index):
        possible_new_atoms = []
        for i in get_possible_connected_new_atom(matrix, indices):
            rel, abs = get_connected_neighbor(matrix, i, indices)
            try:
                tmp_feature = self.feature_dict[mol_index][i]
            except KeyError:
                raise KeyError(f"mol_index: {mol_index}, i: {i}")
            tmp_bond_type = self.bond_matrices[mol_index][abs][i]
            possible_new_atoms.append(
                (
                    i,
                    [tmp_feature, rel, tmp_bond_type],
                ),
            )
        for child in children:
            for idx, possible_new_atom in possible_new_atoms:
                if possible_new_atom == child.atom_features:
                    return child, idx
        return None, None

    def _try_to_add_new_node(self, line, matrix, layer):
        connected_atoms = line["connected_atoms"]
        truth_value = float(line["truth"])
        connected_atom_max_attention = line["connected_atom_max_attention"]
        indices = np.array(connected_atoms)
        mol_index = line["mol_index"]
        try:
            current_node = self._match_atom(layer, line)
            matching_child, new_atom_idx = self._find_matching_child(
                current_node.children,
                matrix,
                indices,
                mol_index,
            )
            if matching_child is not None:
                matching_child.truth_values.append(truth_value)
                matching_child.attention_values.append(float(line["node_attentions"][new_atom_idx]))
                line.iloc[layer] = matching_child.atom_features
                connected_atoms.append(new_atom_idx)
                return matching_child.atom_features
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                rel, abs = get_connected_neighbor(matrix, current_atom_idx, indices)
                new_atom_feature = [
                    self.feature_dict[mol_index][current_atom_idx],
                    rel,
                    self.bond_matrices[mol_index][current_atom_idx][abs],
                ]
                new_node = DevelopNode(
                    atom_features=new_atom_feature,
                    level=layer + 1,
                    truth_values=truth_value,
                    attention_values=connected_atom_max_attention,
                    parent_attention=current_node.parent_attention + np.max(current_node.attention_values),
                )
                current_node.add_child(new_node)
                line.iloc[layer] = new_node.atom_features
                connected_atoms.append(current_atom_idx)
            return new_atom_feature
        except Exception as e:
            raise e

    def _find_matching_child_h(self, children, mol_index, hconnec):
        for child in children:
            possible_new_atom = [self.feature_dict[mol_index][hconnec], -1, -1]

            if possible_new_atom == child.atom_features:
                return child, hconnec
        return None, None

    def _add_node_conn_to_hydrogen(self, line):
        connected_atoms = line["connected_atoms"]
        truth_value = float(line["truth"])
        mol_index = line["mol_index"]
        connected_atom_max_attention = float(line["connected_atom_max_attention"])
        try:
            current_node = self._match_atom(1, line)
            matching_child, new_atom_idx = self._find_matching_child_h(
                current_node.children,
                mol_index,
                line["h_connectivity"],
            )
            if matching_child is not None:
                matching_child.truth_values.append(truth_value)
                matching_child.attention_values.append(float(line["node_attentions"][new_atom_idx]))
                line.iloc[1] = matching_child.atom_features
                connected_atoms.append(new_atom_idx)
                return matching_child.atom_features
            else:
                current_atom_idx = int(line["connected_atom_max_attention_idx"])
                new_atom_feature = [self.feature_dict[mol_index][current_atom_idx], -1, -1]
                new_node = DevelopNode(
                    atom_features=new_atom_feature,
                    level=1 + 1,
                    truth_values=truth_value,
                    attention_values=connected_atom_max_attention,
                    parent_attention=current_node.parent_attention + np.max(current_node.attention_values),
                )
                current_node.add_child(new_node)
                line.iloc[1] = new_node.atom_features
                connected_atoms.append(current_atom_idx)
            return new_atom_feature
        except Exception as e:
            raise e

    def _build_layer_1(self, af: int):
        ci, ca = [], []
        df_work = self.df_af_split[af]
        df_work["total_connected_attention"] = [
            np.sum(row["node_attentions"][row["connected_atoms"]]) for _, row in df_work.iterrows()
        ]
        # df_work = self.df_work.loc[self.df_work["total_connected_attention"] < self.attention_percentage]
        for _, row in df_work.iterrows():
            if row["atomtype"] == "H":
                i, a = (
                    row["h_connectivity"],
                    row["node_attentions"][row["h_connectivity"]],
                )
            else:
                i, a = get_connected_atom_with_max_attention(
                    matrix=self.matrices[int(row["mol_index"])],
                    attentions=np.array(row["node_attentions"]),
                    indices=np.array(row["connected_atoms"]),
                )
            ci.append(i)
            ca.append(a)
        df_work["connected_atom_max_attention_idx"] = ci
        df_work["connected_atom_max_attention"] = ca
        df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
        df_work[1] = [
            self._add_node_conn_to_hydrogen(
                row,
            )
            if row["atomtype"] == "H"
            else self._try_to_add_new_node(
                row,
                self.matrices[row["mol_index"]],
                1,
            )
            for _, row in df_work.iterrows()
        ]

    def _build_tree_single_AF(self, af: int):
        try:
            df_work = self.df_af_split[af]
            self._build_layer_1(af=af)
            # if self.verbose:
            #    print(f"{datetime.datetime.now()}\tAF={af} - Layer {1} done", flush=True)
            for layer in range(2, self.num_layers_to_build):
                try:
                    if self.loggingBuild:
                        self.logger.info(f"\tLayer {layer} started")

                    df_work["total_connected_attention"] = [
                        np.sum(row["node_attentions"][row["connected_atoms"]]) for _, row in df_work.iterrows()
                    ]
                    if self.loggingBuild:
                        self.logger.info("\t\tAttentionSum done")

                    df_work = df_work.loc[df_work["total_connected_attention"] < self.attention_percentage]
                    ai, a = [], []
                    for _, row in df_work.iterrows():
                        x, y = get_connected_atom_with_max_attention(
                            matrix=self.matrices[int(row["mol_index"])],
                            attentions=np.array(row["node_attentions"]),
                            indices=np.array(row["connected_atoms"]),
                        )
                        ai.append(x)
                        a.append(y)
                    if self.loggingBuild:
                        self.logger.info("\t\tMaxAttention done")

                    df_work["connected_atom_max_attention_idx"] = ai
                    df_work["connected_atom_max_attention"] = a
                    df_work = df_work.loc[df_work["connected_atom_max_attention_idx"].notnull()]
                    if df_work.shape[0] == 0:
                        break
                    df_work.sort_values(by="connected_atom_max_attention", ascending=False, inplace=True)
                    if self.loggingBuild:
                        self.logger.info("\t\tSorting done")
                    df_work[layer] = [
                        self._try_to_add_new_node(row, self.matrices[row["mol_index"]], layer)
                        for _, row in df_work.iterrows()
                    ]
                    if self.loggingBuild:
                        self.logger.info("\tAF={af} - Layer {layer} done")
                except Exception as e:
                    print(f"Error in AF {af} - Layer {layer} - {e}")
                    break
                # if self.verbose:
                #    print(f"{datetime.datetime.now()}\tAF={af} - Layer {layer} done", flush=True)
            return self.roots[af]
        except Exception as e:
            print(f"Error in AF {af} - {e}")
            return DevelopNode()

    def build_tree(self, num_processes: int = 6):
        with Pool(num_processes) as p:
            res = p.map(self._build_tree_single_AF, range(AtomFeatures.get_number_of_features()))
        self.root = DevelopNode()
        self.root.children = res
        self.root.update_average()
