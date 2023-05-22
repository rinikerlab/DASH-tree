#!/bin/bash
python -c "from serenityff.charge.gnn.attention_extraction.extractor import Extractor; Extractor._extract_hpc(model='${1}', sdf_index=int(${LSB_JOBINDEX}), scratch='${TMPDIR}')"
mv ${TMPDIR}/${LSB_JOBINDEX}.csv ${2}/.
