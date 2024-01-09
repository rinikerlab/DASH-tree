import numpy as np
import pandas as pd
from serenityff.charge.tree_develop import tree_constructor
from serenityff.charge.gnn.utils.rdkit_helper import get_all_torsion_angles
from serenityff.torsion.tree.tree_utils import get_canon_torsion_feature


class torsion_tree_constructor(tree_constructor):
    def __init__(self, torsion, torsion_type, torsion_type_list, torsion_tree):
        super().__init__(torsion, torsion_type, torsion_type_list, torsion_tree)
        self.df = self.create_torsion_df()

    def create_torsion_df(self):
        df_list = []
        for mol_index, mol in enumerate(self.sdf_suplier):
            torsion_angles_list = get_all_torsion_angles(mol)
            for torsion_indices, torsion_angle in torsion_angles_list:
                # find the four atoms in seld.df and combine them
                a1, a2, a3, a4 = torsion_indices
                df_a1 = self.df[self.df["mol_index"] == mol_index][self.df["atom_index"] == a1].head(1)
                df_a2 = self.df[self.df["mol_index"] == mol_index][self.df["atom_index"] == a2].head(1)
                df_a3 = self.df[self.df["mol_index"] == mol_index][self.df["atom_index"] == a3].head(1)
                df_a4 = self.df[self.df["mol_index"] == mol_index][self.df["atom_index"] == a4].head(1)
                # average node_attentions
                node_attentions = np.mean(
                    [
                        df_a1["node_attentions"].values,
                        df_a2["node_attentions"].values,
                        df_a3["node_attentions"].values,
                        df_a4["node_attentions"].values,
                    ],
                    axis=0,
                )
                new_line = df_a1.copy(deep=True)
                new_line["node_attentions"] = node_attentions
                new_line["truth"] = torsion_angle
                new_line["connected_atoms"].values[0] = [a1, a2, a3, a4]
                af1 = df_a1["atom_feature"].values[0]
                af2 = df_a2["atom_feature"].values[0]
                af3 = df_a3["atom_feature"].values[0]
                af4 = df_a4["atom_feature"].values[0]
                self.df["atom_feature"] = get_canon_torsion_feature(af1, af2, af3, af4)
                df_list.append(new_line)
        df = pd.concat(df_list)
        return df
