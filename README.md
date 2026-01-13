# DGAF-BrainTumorSegmentation

**Official implementation** of Dynamic-Gated Adaptive Feature Fusion (DGAF) for robust multi-modal MRI brain tumor segmentation under **missing/degraded modalities**.

> This repository is the official codebase for our manuscript submitted to **The Visual Computer**:
> **“Enhancing Robustness in Multi-modal Brain Tumor Segmentation via Dynamic-Gated Adaptive Fusion.”**

## 1. Highlights
- **Missing Modality Simulation**: random modality missing/degradation during training to improve robustness.
- **Modality Quality-Aware Scoring**: lightweight quality perception score to estimate modality reliability.
- **Dynamic-Gated Adaptive Fusion**: adaptive fusion weights conditioned on available modalities and quality scores.
- Evaluated on **BraTS2018** and **BraTS2021** with **5-fold cross-validation**.

## 2. Environment & Requirements

### 2.1 Recommended Setup
- OS: Ubuntu 20.04/22.04 or Windows 10/11 (WSL is also ok)
- Python: 3.9–3.11 (recommended 3.10)
- PyTorch: 2.x
- CUDA: 11.x/12.x (depends on your GPU/driver)

### 2.2 Installation (Conda)
conda env create -f environment.yml
conda activate dgaf
pip install -r requirements.txt

### 2.3 Install nnU-Net v2
pip install nnunetv2

## 3. Dataset Preparation (BraTS2018 & BraTS2021)
BraTS datasets are publicly available under their official usage agreements.
We do NOT redistribute any dataset here. Please download BraTS from the official sources and follow the instructions:
See: docs/dataset_preparation.md

### 3.1 nnU-Net Environment Variables
Set nnU-Net paths:

export nnUNet_raw="/path/to/nnUNet_raw"

export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"

export nnUNet_results="/path/to/nnUNet_results"

## 4. Training (5-fold Cross Validation)

<DATASET_ID>: nnU-Net dataset id (e.g., 137，042)

<CONFIG>: 3d_fullres

<TRAINER>: DIMNetTrainer_DGAF

Train 5 folds:

nnUNetv2_train <DATASET_ID> <CONFIG> 0 -tr <TRAINER>

nnUNetv2_train <DATASET_ID> <CONFIG> 1 -tr <TRAINER>

nnUNetv2_train <DATASET_ID> <CONFIG> 2 -tr <TRAINER>

nnUNetv2_train <DATASET_ID> <CONFIG> 3 -tr <TRAINER>

nnUNetv2_train <DATASET_ID> <CONFIG> 4 -tr <TRAINER>

## 5. Inference
nnUNetv2_predict -d <DATASET_ID> -c <CONFIG> -f 0 -tr <TRAINER> -i <INPUT_FOLDER> -o <OUTPUT_FOLDER>

See details: docs/inference.md

## 6. Evaluation
nnU-Net provides built-in evaluation utilities and JSON summaries.

See: docs/evaluation.md

## 7. Reproducibility Notes
We report results using 5-fold cross-validation and average metrics.
Missing/degraded modality simulation is enabled during training (see DGAF trainer/config).
Make sure to document:
GPU, CUDA, PyTorch versions
nnU-Net plans/configuration
random seeds (if used)

## 8. Code & DOI
GitHub: this repository
Zenodo DOI (archived release): TBD (will be added after the first GitHub Release is created)

## 9. Citation
If you find this project useful, please cite our manuscript submitted to The Visual Computer (update will be provided upon publication).

## 10. License
This project is released under the MIT License (see LICENSE).

























