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

trainer = Trainer(device="cuda", loss_function=cross_entropy_loss_for_torsionProfile)
print(trainer.device)


print("start copy")
datafile = "new_training_set170723.pt"
c_datafile = os.environ["TMPDIR"] + "/GNN_lr_%.8f_batch_%i_seed_%i_train.pt" % (lr, batch, seed)
os.system("cp %s %s" % (datafile, c_datafile))
print("copied")
sdf_file = c_datafile
# trainer = Trainer(device = "cuda")
# print(trainer.device)
trainer.model = TorsionWiseAttentiveFP(hidden_channels=200, out_channels=20).to("cuda")
print(trainer.device)
trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=lr)
print(trainer.device)
trainer.save_prefix = "./training/GNN_lr_%.8f_batch_%i_seed_%i_new_set_170823" % (lr, batch, seed)
print("start loading")
trainer.load_graphs_from_pt(pt_file=c_datafile)
# trainer.gen_graphs_from_sdf(sdf_file=sdf_file)
trainer.prepare_training_data(train_ratio=0.9, seed=seed, split_type="smiles")
print("data read in")
epochs = 100
print(trainer.device)
train_loss, eval_loss = trainer.train_model(epochs=epochs, verbose=True, batch_size=batch)


eval_data = trainer.eval_data
eval_data_to_save = []
for entry in eval_data:
    eval_data_to_save.append([entry.smiles, entry.sdf_idx])
pd.DataFrame(eval_data_to_save, columns=["smiles", "sdf_idx"]).to_csv(
    "training/GNN_lr_%.8f_batch_%i_seed_%i_index_new_set.csv" % (lr, batch, seed), index=False
)
