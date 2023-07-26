import pandas as pd
import torch
import argparse
import os

# sys.path.append("/cluster/home/kpaul/whome/projects/serenityff_charge/serenityff-charge")

from serenityff.charge.gnn.training.trainer import Trainer, cross_entropy_loss_for_torsionProfile
from serenityff.charge.gnn.utils.model import TorsionWiseAttentiveFP

parser = argparse.ArgumentParser(description="Run ML MM")
parser.add_argument("-lr", "--lr", type=float, help="learning rate")
parser.add_argument("-b", "--batch", type=int, help="batch to use")
parser.add_argument("-s", "--seed", type=int, help="seed to use")
args = parser.parse_args()


seed = args.seed
lr = args.lr
batch = args.batch


datafile = "sdf_qmugs500_mbis_collect.sdf"
c_datafile = os.environ["TMPDIR"] + "/GNN_lr_%.8f_batch_%i_seed_%i_train.sdf" % (lr, batch, seed)
os.system("cp %s %s" % (datafile, c_datafile))

sdf_file = c_datafile
trainer = Trainer(device="cpu", loss_function=cross_entropy_loss_for_torsionProfile)

trainer.model = TorsionWiseAttentiveFP(hidden_channels=200, out_channels=100)
trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)
trainer.save_prefix = "./training/GNN_lr_%.8f_batch_%ii_seed_%i" % (lr, batch, seed)
# trainer.load_graphs_from_pt(pt_file=pt_file)
trainer.gen_torsion_grahs_from_sdf(sdf_file=sdf_file)

torch.save(trainer._data, "new_training_set170723.pt")

exit()
trainer.prepare_training_data(train_ratio=0.8, seed=seed)

epochs = 500
train_loss, eval_loss = trainer.train_model(epochs=epochs, verbose=True, batch_size=batch)


eval_data = trainer.eval_data
eval_data_to_save = []
for entry in eval_data:
    eval_data_to_save.append([entry.smiles, entry.sdf_idx])
pd.DataFrame(eval_data_to_save, columns=["smiles", "sdf_idx"]).to_csv(
    "training/GNN_lr_%.8f_batch_%i_seed_%i_index.csv" % (lr, batch, seed), index=False
)
