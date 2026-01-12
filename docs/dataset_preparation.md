### docs/dataset_preparation.md
```markdown
# Dataset Preparation (BraTS2018 & BraTS2021)

## 1) Download
BraTS datasets are available via the official BraTS platforms under their respective agreements.
We do NOT redistribute any data.

## 2) Convert to nnU-Net format
Follow nnU-Net v2 dataset format:
- nnUNet_raw/DatasetXXX_YourName/imagesTr
- nnUNet_raw/DatasetXXX_YourName/labelsTr
- nnUNet_raw/DatasetXXX_YourName/dataset.json

## 3) Plan & preprocess
```bash
nnUNetv2_plan_and_preprocess -d <DATASET_ID> --verify_dataset_integrity
