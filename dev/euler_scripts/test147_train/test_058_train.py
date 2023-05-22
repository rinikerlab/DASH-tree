import torch
from rdkit import Chem
import pandas as pd
from serenityff.charge import Trainer, ChargeCorrectedNodeWiseAttentiveFP

print("imports done")
sdf_file = "../test146_combine/combined_multi.sdf"
trainer = Trainer(device="cpu")
trainer.model = ChargeCorrectedNodeWiseAttentiveFP()
trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=10**-5)
trainer.save_prefix = "./test147"
print("trainer init done")
trainer.gen_graphs_from_sdf(sdf_file=sdf_file)
print("gen graph done")
trainer.prepare_training_data(train_ratio=0.8, split_type="smiles")
print("prep train data done")
epochs = 200
train_loss, eval_loss = trainer.train_model(epochs=epochs, verbose=True)

pd.DataFrame({"train_loss": train_loss, "eval_loss": eval_loss}).to_csv("loss.csv", index=False)

# predict on all mols in sdf_file
writer = Chem.SDWriter("predicted.sdf")
for mol in Chem.SDMolSupplier(sdf_file, removeHs=False):
    try:
        mol = trainer.predict(mol)
        writer.write(mol)
    except Exception:
        pass
writer.close()
