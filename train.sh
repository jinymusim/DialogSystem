#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
export TMPDIR=$SCRATCHDIR
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/DialogSystem/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/DialogSystem
python3 train_model.py