# ====== USER CONFIG ======
$DATASET_ID = "<DATASET_ID>"   # e.g., "137"
$CONFIG     = "3d_fullres"
$TRAINER    = "<TRAINER>"      # e.g., "DIMNetTrainer_DGAF"
# =========================

Write-Host "Training 5-fold CV for dataset $DATASET_ID, config $CONFIG, trainer $TRAINER"

0..4 | ForEach-Object {
  $fold = $_
  Write-Host "===== Training fold $fold ====="
  nnUNetv2_train $DATASET_ID $CONFIG $fold -tr $TRAINER
}

Write-Host "Done."
