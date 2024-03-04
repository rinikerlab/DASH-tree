CLEANER_CONTENT = """/
#! /bin/bash

python -c "from serenityff.charge.gnn.attention_extraction.extractor import Extractor;
Extractor._clean_up(num_files=${1}, batch_size=int(${2}), sdf_file='${3}')"
"""
