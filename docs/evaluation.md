
### docs/evaluation.md
```markdown
# Evaluation

nnU-Net provides evaluation summaries in its result folders.
Typically, results and metrics are stored under:
- nnUNet_results/<dataset>/<trainer>/<config>/

You may also use nnU-Net evaluation utilities and parse `summary.json` for DSC/IoU reporting.

Recommended:
- report per-fold results
- compute mean/std across 5 folds
