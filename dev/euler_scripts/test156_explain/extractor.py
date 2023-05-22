# import os
from serenityff.charge.gnn.attention_extraction.extractor import Extractor

# from serenityff.charge import Extractor

# sdf_file = "../test132_train/combined_test.sdf"
# sdf_file = "./sdf_explain.sdf"
sdf_file = "/cluster/work/igc/mlehner/test146_combine/combined_multi.sdf"
# model_path = "./example04_model_sd.pt"
# model_path = "./example05example05"
# model_path = "./example05_model.pt"
model_path = "./GNN_lr_0.00010000_batch_64_seed_1_model_sd.pt"

args = [f"-m {model_path}", f"-s {sdf_file}"]
Extractor.run_extraction_lsf(args)
# Extractor.run_extraction_local(ml_model=model_path, sdf_file=sdf_file, epochs=10)
