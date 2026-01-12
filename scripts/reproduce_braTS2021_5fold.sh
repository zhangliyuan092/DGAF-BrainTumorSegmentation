#!/usr/bin/env bash
set -e

# ====== USER CONFIG ======
DATASET_ID=<DATASET_ID>     # e.g., 137
CONFIG=3d_fullres
TRAINER=<TRAINER>           # e.g., DIMNetTrainer_DGAF
# =========================

echo "Training 5-fold CV for dataset ${DATASET_ID}, config ${CONFIG}, trainer ${TRAINER}"

for FOLD in 0 1 2 3 4
do
  echo "===== Training fold ${FOLD} ====="
  nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER}
done

echo "Done."
