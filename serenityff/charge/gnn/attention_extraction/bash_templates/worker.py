worker_template = """
#! /bin/bash

python -c "from serenityff.charge.gnn.attention_extraction.extractor import Extractor;
Extractor._extract_hpc(model='$1', sdf_index=int(${{ varname }}), scratch='$TMPDIR', sdf_property_name='$3', physics_informed='$4')"

mv ${TMPDIR}/${{{ varname }}}.csv ${2}/.
"""


def get_slurm_worker_content() -> str:
    return worker_template.replace("{{ varname }}", "SLURM_ARRAY_TASK_ID")


def get_lsf_worker_content() -> str:
    return worker_template.replace("{{ varname }}", "LSB_JOBINDEX")
