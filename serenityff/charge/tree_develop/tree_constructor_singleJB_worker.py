import pickle
import os
from typing import Sequence
import argparse

from serenityff.charge.utils import command_to_shell_file
from serenityff.charge.tree.tree_utils import (
    create_new_node_from_develop_node,
)
from serenityff.charge.tree_develop.develop_node import DevelopNode
from serenityff.charge.tree_develop.tree_constructor_parallel_worker import Tree_constructor_parallel_worker


class Tree_constructor_singleJB_worker:
    @staticmethod
    def _parse_filenames(args: Sequence[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--tree_pickle", type=str, required=True)
        parser.add_argument("-a", "--AF_idx", type=int, required=True)
        return parser.parse_args(args)

    @staticmethod
    def build_singleJB(args: Sequence[str]):
        args = Tree_constructor_singleJB_worker._parse_filenames(args)
        tree_pickle = args.tree_pickle
        AF_idx = args.AF_idx
        tree_constructor_parallel_worker = Tree_constructor_parallel_worker()
        root = DevelopNode()
        tree_constructor_parallel_worker = pickle.load(open(tree_pickle, "rb"))
        df_work = tree_constructor_parallel_worker.df_af_split[AF_idx]
        res = tree_constructor_parallel_worker._build_tree_single_AF(af=AF_idx, df_work=df_work)
        root.children.append(res)
        root.update_average()
        new_root = create_new_node_from_develop_node(root)
        del root
        with open(f"{AF_idx}.pkl", "wb") as f:
            pickle.dump(new_root.children[0], f)

    @staticmethod
    def run_singleJB(Tree_constructor_parallel_worker_path: str, AF_idx: int):
        local_tree_constructor = "tree_const_pickle.pkl"
        sub_folder = os.getcwd()
        out_folder = "tree_out"
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        command = f"#SBATCH -n 1\n#SBATCH --cpus-per-task=64\n#SBATCH --time=120:00:00\n#SBATCH --job-name='t_{AF_idx}'\n#SBATCH --nodes=1\n#SBATCH --mem-per-cpu=8000\n#SBATCH --tmp=50000\n#SBATCH --output='t_{AF_idx}.out'\n#SBATCH --error='t_{AF_idx}.err'\n#SBATCH --open-mode=append"
        # copy all the files to the $TMPDIR directory
        command += f"cp {Tree_constructor_parallel_worker_path} $TMPDIR/{local_tree_constructor}"
        command += "cd $TMPDIR"
        command += f"python serenityff/charge/tree_develop/tree_constructor_singleJB_worker.py -p {local_tree_constructor} -a {AF_idx}"
        command += f"cp {AF_idx}.pkl {sub_folder}/tree_out/"
        command_to_shell_file(command, f"singleJB_{AF_idx}.sh")
        os.system(f"sbatch < singleJB_{AF_idx}.sh")
